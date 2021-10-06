import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util


mb = core.Core()
mtu = topo_util.MeshTopoUtil(mb)

material_set_tag = mb.tag_get_handle(
    "MATERIAL_SET", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True
)

coords = np.array(
    [
        [0.000, 0.000, 0.000],  # injection    => 0
        [1.000, 0.000, 0.000],
        [0.000, 1.000, 0.000],
        [1.000, 1.000, 0.000],   # production  => 3
        [0.000, 0.000, 1.000],   # injection   => 4
        [1.000, 0.000, 1.000],
        [0.000, 1.00, 1.000],
        [1.000, 1.000, 1.000],  # production  => 7
        [1.000, 1.000, 0.000],
        [1.000, 0.580, 0.000],
        [0.000, 0.380, 0.000],
        [0.000, 1.000, 0.000],
        [0.000, 0.380, 1.000],
        [1.000, 0.580, 1.000],
        [0.000, 1.000, 1.000],
        [1.000, 1.000, 1.000],
    ]
)

verts = mb.create_vertices(coords.flatten())

# The Lower formation
tetra1 = mb.create_element(
    types.MBTET, [verts[0], verts[1], verts[3], verts[5]]
)
tetra2 = mb.create_element(
    types.MBTET, [verts[0], verts[4], verts[5], verts[6]]
)
tetra3 = mb.create_element(
    types.MBTET, [verts[0], verts[2], verts[3], verts[6]]
)
tetra4 = mb.create_element(
    types.MBTET, [verts[5], verts[6], verts[3], verts[7]]
)
tetra5 = mb.create_element(
    types.MBTET, [verts[0], verts[6], verts[3], verts[5]]
)

all_verts = mb.get_entities_by_dimension(0, 0)
mtu.construct_aentities(all_verts)

formation_volumes = [
    tetra1,
    tetra2,
    tetra3,
    tetra4,
    tetra5
]
formation_volumes_set = mb.create_meshset()
mb.add_entities(formation_volumes_set, formation_volumes)
mb.tag_set_data(material_set_tag, formation_volumes_set, 1)

all_faces = mb.get_entities_by_dimension(0, 2)
neumann_boundary_faces = mb.create_meshset()
for face in all_faces:
    adjacent_vols = mtu.get_bridge_adjacencies(face, 2, 3)
    # print(JI, JK, normal_area_vec)
    if len(adjacent_vols) < 2:
        mb.add_entities(neumann_boundary_faces, [face])
mb.tag_set_data(material_set_tag, neumann_boundary_faces, 201)



injection_wells = [verts[0], verts[4]]
injection_wells_ms = mb.create_meshset()
mb.add_entities(injection_wells_ms, injection_wells)
mb.tag_set_data(material_set_tag, injection_wells_ms, 101)

production_wells = [verts[3], verts[7]]
production_wells_ms = mb.create_meshset()
mb.add_entities(production_wells_ms, production_wells)
mb.tag_set_data(material_set_tag, production_wells_ms, 102)

volumes = mb.get_entities_by_dimension(0, 3)
ms = mb.create_meshset()
mb.add_entities(ms, volumes)

# mb.write_file("test.vtk",)
mb.write_file("meshes/five_vols.h5m")
