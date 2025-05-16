from plyfile import PlyData,PlyElement
import numpy as np

path1 = '65_b.ply'
path2 = '65_q.ply'


ply_base = PlyData.read(path1)
ply_quant = PlyData.read(path2)

vertex_base = np.array(ply_base['vertex'].data.tolist())
vertex_quant = np.array(ply_quant['vertex'].data.tolist())

print("Numero vertici base: ",len(vertex_base) ,"Numero vertiic quant", len(vertex_quant))
#diff = np.abs(vertex_base - vertex_quant)

print("BASE:",ply_base['vertex'].data.dtype)
print("QUANT:",ply_quant['vertex'].data.dtype)


def convert_ply_to_float32(input_path, output_path):
    # Legge il file originale
    plydata = PlyData.read(input_path)
    vertex_data = plydata['vertex'].data

    # Estrai i nomi dei campi
    dtype_old = vertex_data.dtype
    new_dtype = []

    for name in dtype_old.names:
        kind = dtype_old[name].kind
        if name in ('x', 'y', 'z') and kind == 'f':
            new_dtype.append((name, 'f4'))  # forza float32
        else:
            new_dtype.append((name, dtype_old[name]))

    # Crea nuovo array con il nuovo dtype
    new_data = np.empty(len(vertex_data), dtype='f4')

    for name in dtype_old.names:
        new_data[name] = vertex_data[name].astype(newdtype[[n for n,  in new_dtype].index(name)][1])

    # Crea nuovo elemento e scrivi file
    new_el = PlyElement.describe(new_data, 'vertex')
    PlyData([new_el], text=False).write(output_path)

    print(f"✅ Salvato: {output_path} (convertito a float32)")

# convert_ply_to_float32('65_q.ply', 'ottimizzato.ply')
#
# ply_opt = PlyData.read('ottimizzato.ply')
# vertex_opt= np.array(ply_opt['vertex'].data.tolist())
# print("opt:",len(vertex_opt))

