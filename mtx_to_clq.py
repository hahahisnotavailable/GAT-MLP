import os


def convert_mtx_to_clq(mtx_file_path, clq_file_path):

    try:
        with open(mtx_file_path, "r") as fin, open(clq_file_path, "w") as fout:
            edge_count = 0
            node_count = 0
            parsing_edges = False

            for line in fin:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                if not parsing_edges:

                    try:
                        parts = list(map(int, line.split()))
                        if len(parts) != 3:
                            print(f"incorrectly formatted: {line}")
                            return False
                        node_count = parts[0]
                        edge_count = parts[2]
                        fout.write(f"p edge {node_count} {edge_count}\n")
                        parsing_edges = True
                    except:
                        print(f"Size information cannot be resolved: {line}")
                        return False
                else:
                    try:
                        i, j = map(int, line.split()[:2])
                        fout.write(f"e {i} {j}\n")
                    except:
                        continue

        return True
    except Exception as e:
        print(f"File processing failed {mtx_file_path}: {e}")
        return False


def batch_convert_mtx_to_clq(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mtx"):
            mtx_file_path = os.path.join(input_folder, file_name)
            clq_file_path = os.path.join(output_folder, file_name.replace(".mtx", ".clq"))

            if os.path.exists(clq_file_path):
                print(f"Skip the converted files: {file_name}")
                continue

            success = convert_mtx_to_clq(mtx_file_path, clq_file_path)
            if success:
                print(f"Completed: {file_name}")
                try:
                    os.remove(mtx_file_path)
                    print(f"Deleted: {file_name}")
                except Exception as e:
                    print(f" {file_name}: {e}")
            else:
                print(f"File processing failed {file_name}")



input_folder = input("Enter the folder path where the .mtx files are located: ")
output_folder = input("Enter the folder path to save the .clq files (it will be created if doesn't exist): ")
batch_convert_mtx_to_clq(input_folder, output_folder)

