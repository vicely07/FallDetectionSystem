wget -N --content-disposition http://vlg.cs.dartmouth.edu/c3d/conv3d_deepnetA_sport1m_iter_1900000 --directory-prefix=${DIR}
wget -N https://raw.githubusercontent.com/gtoderici/sports-1m-dataset/master/labels.txt --directory-prefix=${DIR} 
wget https://raw.githubusercontent.com/facebook/C3D/master/C3D-v1.0/src/caffe/proto/caffe.proto
sudo apt-get install protobuf-compiler
protoc --python_out=. caffe.proto
