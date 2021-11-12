
#wget https://github.com/danbarto/snowblower/archive/master.zip
#unzip master.zip
#mv snowblower-master snowblower

mkdir -p snowblower/tools/
cp -r tools/*.py snowblower/tools/

tar -czf snowblower.tar.gz snowblower

rm -rf snowblower

mv snowblower.tar.gz tools/analysis.tar.gz
