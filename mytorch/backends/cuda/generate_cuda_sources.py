import os.path as osp
from mytorch.backends.cuda.env import CudaEnv

if __name__ == "__main__":
    folder = osp.join(osp.dirname(__file__), "../../native/cuda")
    for filename in ["conv.cu", "batch_norm.cu"]:
        if osp.isdir(osp.join(folder, filename)) and filename.endswith(".cu"):
            CudaEnv.instance().compiler.save_templated_source(filename)
