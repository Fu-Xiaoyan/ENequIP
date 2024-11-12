import ase
import ase.io
import numpy as np
import ase.neighborlist
import ase
from ase.calculators.singlepoint import SinglePointCalculator, SinglePointDFTCalculator
from ase.calculators.calculator import all_properties as ase_all_properties
from copy import deepcopy
from dataset.torch_geometric import Data

from torch import tensor as Tensor
import torch
int_type = torch.int64
float_type = torch.float32
bool_type = torch.bool


class AtomicData(Data):
    """A neighbor graph for points in (periodic triclinic) real space.

    For typical cases either ``from_points`` or ``from_ase`` should be used to
    construct a AtomicData; they also standardize and check their input much more
    thoroughly.

    In general, ``node_features`` are features or input information on the nodes that will be fed through and transformed by the network, while ``node_attrs`` are _encodings_ fixed, inherant attributes of the atoms themselves that remain constant through the network.
    For example, a one-hot _encoding_ of atomic species is a node attribute, while some observed instantaneous property of that atom (current partial charge, for example), would be a feature.

    In general, ``torch.Tensor`` arguments should be of consistant dtype. Numpy arrays will be converted to ``torch.Tensor``s; those of floating point dtype will be converted to ``torch.get_current_dtype()`` regardless of their original precision. Scalar values (Python scalars or ``torch.Tensor``s of shape ``()``) a resized to tensors of shape ``[1]``. Per-atom scalar values should be given with shape ``[N_at, 1]``.

    ``AtomicData`` should be used for all data creation and manipulation outside of the model; inside of the model ``AtomicDataDict.Type`` is used.

    Args:
        pos (Tensor [n_nodes, 3]): Positions of the nodes.
        edge_index (LongTensor [2, n_edges]): ``edge_index[0]`` is the per-edge
            index of the source node and ``edge_index[1]`` is the target node.
        edge_cell_shift (Tensor [n_edges, 3], optional): which periodic image
            of the target point each edge goes to, relative to the source point.
        cell (Tensor [1, 3, 3], optional): the periodic cell for
            ``edge_cell_shift`` as the three triclinic cell vectors.
        node_features (Tensor [n_atom, ...]): the input features of the nodes, optional
        node_attrs (Tensor [n_atom, ...]): the attributes of the nodes, for instance the atom type, optional
        batch (Tensor [n_atom]): the graph to which the node belongs, optional
        atomic_numbers (Tensor [n_atom]): optional.
        atom_type (Tensor [n_atom]): optional.
        **kwargs: other data, optional.
    """

    def __init__(self, **kwargs):
        kwargs['forces'] = Tensor(kwargs.get('forces', 0), dtype=float_type)
        kwargs['total_energy'] = Tensor(kwargs.get('total_energy', 0), dtype=float_type)
        kwargs['free_energy'] = Tensor(kwargs.get('free_energy', 0), dtype=float_type)
        kwargs['atomic_numbers'] = Tensor(kwargs['atomic_numbers'], dtype=int_type)
        kwargs['E_field'] = Tensor(kwargs.get('E_field', 0), dtype=float_type)
        kwargs['vaccum'] = Tensor(kwargs.get('vaccum', 0), dtype=float_type)
        kwargs['fermi'] = Tensor(kwargs.get('fermi', 0), dtype=float_type)
        super().__init__(num_nodes=len(kwargs["pos"]), **kwargs)

    @classmethod
    def from_ase(
        cls,
        atoms,
        r_max,
        **kwargs,
    ):
        """Build a ``AtomicData`` from an ``ase.Atoms`` object.

        Respects ``atoms``'s ``pbc`` and ``cell``.

        First tries to extract energies and forces from a single-point calculator associated with the ``Atoms`` if one is present and has those fields.
        If either is not found, the method will look for ``energy``/``energies`` and ``force``/``forces`` in ``atoms.arrays``.

        `get_atomic_numbers()` will be stored as the atomic_numbers attribute.

        Args:
            atoms (ase.Atoms): the input.
            r_max (float): neighbor cutoff radius.
            features (torch.Tensor shape [N, M], optional): per-atom M-dimensional feature vectors. If ``None`` (the
             default), uses a one-hot encoding of the species present in ``atoms``.
            include_keys (list): list of additional keys to include in AtomicData aside from the ones defined in
                 ase.calculators.calculator.all_properties. Optional
            key_mapping (dict): rename ase property name to a new string name. Optional
            **kwargs (optional): other arguments for the ``AtomicData`` constructor.

        Returns:
            A ``AtomicData``.
        """

        assert "pos" not in kwargs

        default_args = set(['numbers', 'pos', 'pbc', 'r_max', 'positions', 'cell'])
        # the keys that are duplicated in kwargs are removed from the include_keys
        include_keys = [ 'forces', 'energy', 'energies', 'charges', 'dipole', 'free_energy', 'initial_magmoms', 'vaccum', 'fermi']

        key_mapping = {'forces': 'forces', 'energy': 'total_energy', 'initial_magmoms': 'E_field'}

        add_fields = {}
        # Get info from atoms.arrays; lowest priority. copy first
        add_fields = {
            key_mapping.get(k, k): v
            for k, v in atoms.arrays.items()
            if k in include_keys
        }
        # Get info from atoms.info; second lowest priority.
        add_fields.update(
            {
                key_mapping.get(k, k): v
                for k, v in atoms.info.items()
                if k in include_keys
            }
        )

        if atoms.calc is not None:

            if isinstance(
                atoms.calc, (SinglePointCalculator, SinglePointDFTCalculator)
            ):
                add_fields.update(
                    {
                        key_mapping.get(k, k): deepcopy(v)
                        for k, v in atoms.calc.results.items()
                        if k in include_keys
                    }
                )

        add_fields['atomic_numbers'] = atoms.get_atomic_numbers()

        # cell and pbc in kwargs can override the ones stored in atoms
        cell = kwargs.pop("cell", atoms.get_cell())
        pbc = kwargs.pop("pbc", atoms.pbc)

        return cls.from_points(
            pos=atoms.positions,
            r_max=r_max,
            cell=cell,
            pbc=pbc,
            **kwargs,
            **add_fields,
        )

    @classmethod
    def from_points(
        cls,
        pos=None,
        r_max: float = None,
        self_interaction: bool = False,
        strict_self_interaction: bool = True,
        cell=None,
        pbc=None,
        **kwargs,
    ):
        """Build neighbor graph from points, optionally with PBC.

        Args:
            pos (np.ndarray/torch.Tensor shape [N, 3]): node positions. If Tensor, must be on the CPU.
            r_max (float): neighbor cutoff radius.
            cell (ase.Cell/ndarray [3,3], optional): periodic cell for the points. Defaults to ``None``.
            pbc (bool or 3-tuple of bool, optional): whether to apply periodic boundary conditions to all or each of
            the three cell vector directions. Defaults to ``False``.
            self_interaction (bool, optional): whether to include self edges for points. Defaults to ``False``. Note
            that edges between the same atom in different periodic images are still included. (See
            ``strict_self_interaction`` to control this behaviour.)
            strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if the
            two instances of the atom are in different periodic images. Defaults to True, should be True for most
            applications.
            **kwargs (optional): other fields to add. Keys listed in ``AtomicDataDict.*_KEY` will be treated specially.
        """
        if pos is None or r_max is None:
            raise ValueError("pos and r_max must be given.")

        if pbc is None:
            if cell is not None:
                raise ValueError(
                    "A cell was provided, but pbc weren't. Please explicitly probide PBC."
                )
            # there are no PBC if cell and pbc are not provided
            pbc = False

        if isinstance(pbc, bool):
            pbc = (pbc,) * 3
        else:
            assert len(pbc) == 3

        edge_index, edge_cell_shift, cell = neighbor_list_and_relative_vec(
            pos=pos,
            r_max=r_max,
            self_interaction=self_interaction,
            strict_self_interaction=strict_self_interaction,
            cell=cell,
            pbc=pbc,
        )
        pos = Tensor(pos, dtype=float_type)
        # Make torch tensors for data:
        if cell is not None:
            kwargs['cell'] = cell.view(3, 3)
            kwargs['edge_cell_shift'] = edge_cell_shift
        if pbc is not None:
            kwargs['pbc'] = Tensor(
                pbc, dtype=bool_type
            ).view(3)

        return cls(edge_index=edge_index, pos=pos, **kwargs)


def neighbor_list_and_relative_vec(
    pos,
    r_max,
    self_interaction=False,
    strict_self_interaction=True,
    cell=None,
    pbc=False,
):
    """Create neighbor list and neighbor vectors based on radial cutoff.

    Create neighbor list (``edge_index``) and relative vectors
    (``edge_attr``) based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`

    If the input positions are a tensor with ``requires_grad == True``,
    the output displacement vectors will be correctly attached to the inputs
    for autograd.

    All outputs are Tensors on the same device as ``pos``; this allows future
    optimization of the neighbor list on the GPU.

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions. Ignored if ``pbc == False``.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the three cell dimensions.
        self_interaction (bool): Whether or not to include same periodic image self-edges in the neighbor list.
        strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if the two
            instances of the atom are in different periodic images. Defaults to True, should be True for most applications.

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_cell_shift (torch.tensor shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
        cell (torch.Tensor [3, 3]): the cell as a tensor on the correct device.
            Returned only if cell is not None.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3
    temp_pos = pos
    # don't know what will happen if all use gpu
    # Get a cell on the CPU no matter what
    if cell is not None:
        temp_cell = np.asarray(cell)
        cell_tensor = Tensor(temp_cell, dtype=float_type)
    else:
        # ASE will "complete" this correctly.
        temp_cell = np.zeros((3, 3), dtype=np.float32)
        cell_tensor = Tensor(temp_cell, dtype=float_type)

    # ASE dependent part
    temp_cell = ase.geometry.complete_cell(temp_cell)

    first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=float(r_max),
        self_interaction=strict_self_interaction,  # we want edges from atom to itself in different periodic images!
        use_scaled_positions=False,
    )

    # Eliminate true self-edges that don't cross periodic boundaries
    if not self_interaction:
        bad_edge = first_idex == second_idex
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise ValueError(
                "After eliminating self edges, no edges remain in this system."
            )
        first_idex = first_idex[keep_edge]
        second_idex = second_idex[keep_edge]
        shifts = shifts[keep_edge]

    # Build output:
    edge_index = Tensor(np.vstack(
        (first_idex, second_idex)
    ), dtype=int_type)

    shifts = Tensor(
        shifts,
        dtype=float_type,
    )
    return edge_index, shifts, cell_tensor

def get_property(object):
    property_list = []
    for p in dir(object):
        if '__' not in p:
            property_list.append(p)
    tensor_property = []
    for p in property_list:
        attr = getattr(object, p)
        if type(getattr(object, p)) == torch.Tensor:
            tensor_property.append(p)
    return tensor_property


def ase_dataset_reader(
    ase_kwargs: dict,
    atomicdata_kwargs: dict,
    device=None):
    datas = []
    # stream them from ase too using iread
    for i, atoms in enumerate(ase.io.iread(**ase_kwargs, index=slice(0, None, 1), parallel=False)):
        datas.append(AtomicData.from_ase(atoms=atoms, **atomicdata_kwargs))
    if device:
        properties = get_property(datas[0])
        for data in datas:
            for p in properties:
                tensor_ = getattr(data, p)
                setattr(data, p, tensor_.to(device=device))
    return datas

if __name__ == '__main__':
    ase_kwargs = {'filename': 'test-E.extxyz', 'format': 'extxyz'}
    atomicdata_kwargs = {'r_max': 4.0}
    ase_dataset_reader(ase_kwargs, atomicdata_kwargs)