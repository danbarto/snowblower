# snowblower
Tools for the Snowmass process.

To setup:
```
git clone --recursive git@github.com:danbarto/snowblower.git

cd snowblower
cmsrel CMSSW_10_6_19
cd CMSSW_10_6_19/src/
cmsenv
git cms-init
cmsenv
scram b -j 8
cd ../../
```


To use the sample caches:
```
pip install klepto --user
```

## Conda instructions

### Installing conda

``` shell
curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b 
```

Miniconda should make modifications to your `.bashrc`. Make sure that this is the case, and if you're working on the uaf
add `source ~/.bashrc` in your `.profile`.

Re-login to the uaf and run the following commands:
``` shell
~/miniconda3/bin/conda init
conda config --set auto_activate_base false
conda config --add channels conda-forge
```

If you plan to use DASK you'll need conda-pack:
``` shell
conda install --name base conda-pack -y
```

### Local conda environment

```shell
conda create --name coffeadev4 uproot dask dask-jobqueue matplotlib pandas jupyter hdfs3 pyarrow fastparquet numba numexpr coffea -y
pip install yahist klepto onnxruntime sklearn lbn
```


### DASK worker environment

``` shell
conda deactivate
conda create --name workerenv uproot dask dask-jobqueue pyarrow fastparquet numba numexpr boost-histogram onnxruntime -y
conda run --name workerenv pip install coffea yahist
conda activate workerenv
```

then pack it

``` shell
conda pack -n workerenv --arcroot workerenv -f --format tar.gz --compress-level 9 -j 8 --exclude "*.pyc" --exclude "*.js.map" --exclude "*.a"
```
and move it to `tools/`

## Running DASK

Pack the code with: `source packCode.sh`. Make sure you have a valid proxy and have the workerenv tarball in `tools/`.
Run `ipython -i start_cluster.py -- --scale 50` for 50 workers. Don't kill the ipython session or the cluster will also shutdown again.
To test, run `ipython -i test_dask.py`.
A coffea processor can be run in `analysis/` with `ipython -i test.py -- --run_flat_remote --dask`. This should take O(5mins) for 10-50 active workers.
