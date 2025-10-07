---
sidebar_position: 4
---

# suitesparse-dl
> A tool to download matrices from SuiteSparse Matrix Collection(https://sparse.tamu.edu) site.

## Build
```bash
go build
```

## How to use
```bash
./suitesparse fetch # fetch metadata
mkdir -p dl/1k dl/10k dl/100k dl/1M dl/10M dl/100M dl/1G dl/10G # create directories   
./suitesparse-dl dl
```

## Extract Matrix Market from .tar.gz
We can use following srcipt to extract matrix market file from downloaded .tar.gz file.
```bash
#!/bin/bash

root_dir='dl/100k'
target_dir='dl_mm/100k'
mkdir -p $target_dir
find $root_dir -name "*tar.gz" | sort > tar_list.txt
for tar_path in `cat tar_list.txt`; do
    tar -zxvf $tar_path -C $target_dir --strip-components 1 > /dev/null
done
rm tar_list.txt
```

## Special note for matrices with the same name
The matrix name is not the unique key in SuiteSparse Matrix Collection.
But suitsparse-dl use matrix name as filename (key).
Thus, matrices with the same name will only be downloaded once when the name first apprears.
In other words, if 2 or more matrices with the same name, we only download the first matrix, other matrices will be ignored.

A workaround for this problem is: user can **manually** download the matrices with the same name and give them different filenames.

The matrices with the same name are list as following:

| Name         | IDs |
| ------------ | --- |
| nasa1824     | 363, 757  |
| nasa2910     | 364, 759  |
| nasa4704     | 365, 760  |
| barth        | 754, 865  |
| barth4       | 755, 866  |
| barth5       | 756, 867  |
| pwt          | 762, 880, 1273 |
| shuttle_eddy | 763, 881  |
| skirt        | 764, 882  |
| copter2      | 1230, 1256|
| ex3sta1      | 1379, 1709 (*) |
| pf2177       | 1394, 1753 (*) |
| fxm3_6       | 1380, 1805 (*) |
| fxm4_6       | 1381, 1807 (*) |
| football     | 1474, 2397 (*) |

## Generate a sbatch file for job submitting
suitsparse-dl support to generate a sbatch file for job submitting 
(e.g. the case of running benchmark on [slurm](https://slurm.schedmd.com/overview.html) Workload Manager system).

You can run following command to generate a sbatch file:
```bash
./suitesparse-dl gen --data ./dl_mm --output spmv_batch.sh --tpl template.sh
```
where, `--data` point to the path of matrices, `--output` specific the output sbatch file and `--tpl` can specific the template file. 
For information, can run `./suitesparse-dl gen -h`.  
Note: `--data` is a path to parent directory of matrix directories,
and the matrix directory (e.g. directory `08blocks`) should keep the same name as the matrix file name within it.
Following shows an example of the layout of data directory `./dl_mm`.
```log
./dl_mm/
├── 08blocks
│         └── 08blocks.mtx
├── adjnoun
│         ├── adjnoun.mtx
├── ash219
│         └── ash219.mtx
├── ash331
│         └── ash331.mtx
└── ash85
          └── ash85.mtx
```

If you want to generate from bin2 file, you can specific flag `--type bin2`, `--data` points to the parent dir of .bin2 file.
```bash
./suitesparse-dl gen --data ./bin2/ --output spmv_batch.sh --tpl template.sh --type bin2
```

## Workflow to generating batch script
```bash
suitesparse-dl fetch # fetch metadata
suitesparse-dl dl # download tar.gz of each matrix
# below we take 100k as example: tar.gz is saved at ./dl; .mtx is saved at ./dl_mm; .bin2 is saved at .bin2.
./extract.sh # extract from .tar.gz (without --strip-components 1 to tar command)
suitesparse-dl list -d ./dl_mm/100k/ > 100k.list
suitesparse-dl conv -b -mm ./100k.list -o ./bin2/100k/
suitesparse-dl gen --data ./bin2/100k/ --output spmv_batch.sh --tpl template.sh --type bin2
```

## .bin2 binary format
`suitesparse-dl conv` can convert .mtx format to a customized binary file format .bin2.  
In .bin2 file, the data format is:
```text
| magic number (4 bytes)   | binary format (4 bytes)    | data type (4 bytes) |
| number of rows (4 bytes) | number of column (4 bytes) | nnz (4 bytes)        |
| CSR row_offset array | CSR col_index array | CSR values array |
```

For each part:
- **magic number**: it is unsigned int (4 bytes) and should always equals to `0x20211015`.
- **binary format**: the file format version (4 bytes). Current is version 2 (it should always equals to `0x2`).
- **data type**: the data type in the sparse matrix (4 bytes). It can be: 1 (boolean), 2 (integer), 3 (real) and 4(complex).
- **number of rows**: signed int (4 bytes).
- **number of column**: signed int (4 bytes).
- **nnz**: signed int (4 bytes).
- **CSR `row_offset` array**: It is an array and contains (number of rows + 1) elements. Each element is signed int (4 bytes).
- **CSR `col_index` array**: It is an array and contains (number of column) elements. Each element is signed int (4 bytes).
- **CSR `values` array**: If the data type is integer (`int` in cpp, 4 bytes) or real (`double` in cpp, 8 bytes), it is an array and contains nnz elements). If the data type is boolean, it containe 0 element (empty array). Complex is not supported now (when converting from .mtx to .bin2 format, suitesparse-dl only convert the real part and result in an array of real data type).
  
An reference code for reading .bin2 file can be found at [cli/matrix_format/csr_binary_reader.hpp#load_mat()](https://github.com/usparse/usparse/blob/main/cli/matrix_format/csr_binary_reader.hpp).
