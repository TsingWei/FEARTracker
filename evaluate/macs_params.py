import torch
import os
import sys
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
# from fire import Fire
from hydra.utils import instantiate
from thop import profile
from thop.utils import clever_format
import time
import numpy as np
import onnxruntime as ort
from onnxconverter_common import float16
import onnx
from onnxsim import simplify as simplify_func
from model_training.utils.hydra import load_yaml
import os
# os.environ["http_proxy"] = "http://10.21.59.42:10809"
# os.environ["https_proxy"] = "http://10.21.59.42:10809"

class ProfileTrackingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, search, template):
        pred = self.model.track(search, template)
        return [pred["TARGET_REGRESSION_LABEL_KEY"], pred["TARGET_CLASSIFICATION_KEY"]]


def main(config_path: str = "model_training/config/model/fear.yaml"):
    config = load_yaml(config_path)
    model = instantiate(config)
    model = ProfileTrackingWrapper(model)

    search_inp = torch.rand(1, 3, 256, 256)
    template_inp = torch.rand(1, 256, 8, 8)
    # macs, params = profile(model, inputs=(search_inp, template_inp), custom_ops=None, verbose=False)
    # macs, params = clever_format([macs, params], "%.3f")
    # print('overall macs is ', macs)
    # print('overall params is ', params)

    output_onnx_name = 'test_net.onnx'
    
    dtype = np.float32
    inputs=(search_inp, template_inp)
    inputs_onnx = {'x':  np.array(search_inp.cpu(), dtype=dtype),
                   'z': np.array(template_inp.cpu(), dtype=dtype),
                #    't
                    }
    
    # torch.onnx.export(model, 
    #     inputs,
    #     output_onnx_name, 
    #     input_names=[ "x", "z"], 
    #     output_names=["o1","o2"],
    #     opset_version=11,
    #     export_params=True,
    #     # verbose=True,
    #     # dynamic_axes={'input':{0:'batch', 2:'h', 3:'w'}, 'output':{0:'batch', 2:'h2', 3:'w2'}} 
    # )

    # providers = ['CUDAExecutionProvider']
    # onnx_model = onnx.load("test_net.onnx")
    
    # # model_fp16 = float16.convert_float_to_float16(model)
    # model_fp16, success = simplify_func(onnx_model)
    # # model_fp16 = float16.convert_float_to_float16(model_fp16)
    # assert success
    # onnx.save(model_fp16, "test_net_fp16.onnx")
    # ort_session = ort.InferenceSession("test_net_fp16.onnx", providers=providers)
    # # output = ort_session.run(output_names=['output'],
    # #                          	input_feed=inputs_onnx,
    # #                             )


    T_w = 100  # warmup
    T_t = 500  # test
    with torch.no_grad():
        for i in range(T_w):
            oup = model(search_inp, template_inp)
            # output = ort_session.run(output_names=["o1","o2"],
            #                  	input_feed=inputs_onnx,
            #                     )
        t_s = time.time()
        for i in range(T_t):
            oup = model(search_inp, template_inp)
            # output = ort_session.run(output_names=["o1","o2"],
            #                  	input_feed=inputs_onnx,
            #                     )
        torch.cuda.synchronize()
        t_e = time.time()
    print('speed: %.2f FPS' % (T_t / (t_e - t_s)))


if __name__ == '__main__':
    main()
