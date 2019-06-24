#
# Install CUDA.
#
# Version is given in the CUDA variable. If left empty, this script does
# nothing. As variables are exported by this script, "source" it rather than
# executing it.
#
# Known versions:
#  CUDA=5.5  => CUDA_REPO=ubuntu1404 CUDA_VERSION=5.5-54
#  CUDA=6.5  => CUDA_REPO=ubuntu1404 CUDA_VERSION=6.5-19
#  CUDA=7.0  => CUDA_REPO=ubuntu1404 CUDA_VERSION=7.0-28
#  CUDA=7.5  => CUDA_REPO=ubuntu1404 CUDA_VERSION=7.5-18
#  CUDA=8.0  => CUDA_REPO=ubuntu1404 CUDA_VERSION=8.0.61-1
#  CUDA=9.0  => CUDA_REPO=ubuntu1604 CUDA_VERSION=9.0.176-1
#  CUDA=9.1  => CUDA_REPO=ubuntu1604 CUDA_VERSION=9.1.85-1

if [ -n "$CUDA" ]; then
    case "$CUDA" in
	"5.5")
	    CUDA_REPO=ubuntu1404
	    CUDA_VERSION=5.5-54
	    CUDA=5.5-power8
	    ;;
	"6.5")
	    CUDA_REPO=ubuntu1404
	    CUDA_VERSION=6.5-19
	    ;;
	"7.0")
	    CUDA_REPO=ubuntu1404
	    CUDA_VERSION=7.0-28
	    ;;
	"7.5")
	    CUDA_REPO=ubuntu1404
	    CUDA_VERSION=7.5-18
	    ;;
	"8.0")
	    CUDA_REPO=ubuntu1404
	    CUDA_VERSION=8.0.61-1
	    ;;
	"9.0")
	    CUDA_REPO=ubuntu1604
	    CUDA_VERSION=9.0.176-1
	    ;;
	"9.1")
	    CUDA_REPO=ubuntu1604
	    CUDNN_VERSION 7.1.2.21
	    CUDA_VERSION=9.1.85-1
	    NCCL_VERSION=2.1.15
	    CUDA_CUDNN_APT=libcudnn7=$CUDNN_VERSION-1+cuda9.1 libcudnn7-dev=$CUDNN_VERSION-1+cuda9.1
	    CUDA_EXTRA_APT=libnccl-dev=$NCCL_VERSION-1+cuda9.1 
	    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5
	    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80
	    ;;
	*)
	    echo "Unsupported CUDA version $CUDA."
	    exit 1
    esac

    echo "Installing CUDA support"
    if [ "$CUDA_REPO" == "ubuntu1604" ]; then
    	travis_retry sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
	travis_retry sudo apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub
	travis_retry sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
	sudo echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub
    fi
    
    sudo echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    sudo echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

    travis_retry sudo apt-get update -qq
    export CUDA_APT=${CUDA/./-}

    travis_retry sudo apt-get install -y cuda-command-line-tools-${CUDA_APT} ${CUDA_CUDNN_APT} ${CUDA_EXTRA_APT}
    travis_retry sudo apt-get clean

    sudo echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf
    sudo echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

    export CUDA_HOME=/usr/local/cuda-${CUDA}
    sudo ln -s ${CUDA_HOME} /usr/local/cuda
    export LD_LIBRARY_PATH=${CUDA_HOME}/nvvm/lib64:/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/cuda/lib64/${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:/usr/local/cuda/bin:${PATH}

    nvcc --version
else
    echo "CUDA is NOT installed."
fi
