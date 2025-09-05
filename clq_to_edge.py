import os



def convert_clq_to_edge_list(clq_file_path, output_dir):
    with open(clq_file_path, 'r') as file:
        lines = file.readlines()


    vertices = set()
    edges = []


    for line in lines:
        if line.startswith('p'):

            _, _, vertex_count, edge_count = line.strip().split()
            vertex_count = int(vertex_count)
            edge_count = int(edge_count)
        elif line.startswith('e'):

            _, u, v = line.strip().split()
            u, v = int(u), int(v)


            u -= 1
            v -= 1

            vertices.add(u)
            vertices.add(v)
            edges.append((u, v))


    output_file_name = os.path.basename(clq_file_path).replace('.clq', '.txt')
    output_file_path = os.path.join(output_dir, output_file_name)


    with open(output_file_path, 'w') as out_file:
        out_file.write(f"{len(vertices)} {len(edges)}\n")
        for u, v in edges:
            out_file.write(f"{u} {v}\n")

    print(f"File has been saved to: {output_file_path}")



def convert_clq_files_in_directory(folder_path, output_dir):

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.clq'):
            clq_file_path = os.path.join(folder_path, file_name)
            convert_clq_to_edge_list(clq_file_path, output_dir)



folder_path = input("Enter the folder path containing the .clq files: ")
output_dir = input("Enter the output folder path (leave empty to save in the same folder): ")


if not output_dir:
    output_dir = folder_path

convert_clq_files_in_directory(folder_path, output_dir)
