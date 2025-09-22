import os.path as osp
from mytorch.backends.cuda.env import CudaSourceGenerator

if __name__ == "__main__":
    generator = CudaSourceGenerator()
    folder = osp.join(osp.dirname(__file__), "../../native/cuda")
    for filename in ["conv.cu", "batch_norm.cu"]:
        if osp.isdir(osp.join(folder, filename)) and filename.endswith(".cu"):
            generator.save_templated_source(filename)
