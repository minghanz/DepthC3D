import zipfile
import re
import os

##### This script is to extract new kitti depth dataset (with denser ground truth as depth images) into the corresponding folders of sequences
##### This includes train and val data.
##### The test_files.txt in engen_benchmark split mostly corresponds to files in val folder in this zip, except that 09_28 images are in train set. 

file_name = "data_download/data_depth_annotated.zip"
target_folder = "kitti_data"

zip_obj = zipfile.ZipFile(file_name)
list_of_files = zip_obj.namelist()

not_found_seqs = []
for i,f in enumerate(list_of_files):
    # print(f)
    result = re.search("/", f)
    str_needed = f[result.end():]
    # print(str_needed)
    date = str_needed[:10]
    seq_folder = str_needed.split("/")[0]
    seq_path = os.path.join(target_folder, date, seq_folder)

    if os.path.exists(seq_path):
        target_path = os.path.join(target_folder, date, str_needed)
        zip_obj.extract(f, path=target_path)
        if i%100==0:
            print(i, "---------------\n", f, "\n", target_path, "\n")
    else:
        if seq_folder not in not_found_seqs:
            not_found_seqs.append(seq_folder)
            # print("seq {} not found!".format(seq_folder))
            # break
    # break
    # print("----------------")
with open("data_download/readme.txt", "w") as f:
    f.writelines("%s\n"%seq for seq in not_found_seqs)
# print(not_found_seqs)

# with file_bj.open() as f:
#     zip_obj.extract("subfile", path="target")
#     zip_obj.namelist()