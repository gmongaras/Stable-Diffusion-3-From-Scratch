import pickle
import torch
from torch import serialization
import torch_tensorrt
from transformers import AutoModelForSeq2SeqLM
import os
from transformers import CLIPModel

os.environ["TORCH_DYNAMO_MULTI_GPU_SAFE"] = "1"

# Set default pickle protocol to 4
pickle.DEFAULT_PROTOCOL = 4
torch.serialization.DEFAULT_PROTOCOL = 4
serialization.DEFAULT_PROTOCOL = 4


# Compile model
def compile_model(model_, out_name):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.model = model_
            # Turn off dropout, etc
            for param in self.model.parameters():
                param.requires_grad = False
            # Iterate over all modules and set dropout probability to 0
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0
        @torch.no_grad()
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    model = Model().eval().cuda()

    # Examples. Make sure batch isn ot 1
    input_ids_example = torch.randint(1, 128, (2, 77), device="cuda").int()
    attn_mask_example = torch.ones(2, 77, device="cuda").bool()
    inputs_ = [
            input_ids_example,
            attn_mask_example
    ]

    # Batch size can be from 1 to 512
    batchsize = torch.export.Dim("batchsize", min=1, max=512)
    dynamic_shapes=({0: batchsize}, {0: batchsize})
    exp_program = torch.export.export(model, tuple(inputs_), dynamic_shapes=dynamic_shapes)
    trt_gm = torch_tensorrt.dynamo.compile(exp_program, inputs=inputs_)
    trt_gm(*inputs_)
    ### NOTE: For now if this gives a serialization error, you have to manually change
    ###       torch.serialization.DEFAULT_PROTOCOL to 4 (like edit this in the code on your computer)
    torch_tensorrt.save(trt_gm, f"./models/{out_name}.ep", inputs=inputs_) # PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file
    torch_tensorrt.save(trt_gm, f"./models/{out_name}.ts", output_format="torchscript", inputs=inputs_)
    del model











    # Testing if it works

    inputs_ = [
        torch.randint(1, 128, (96*3, 77), device="cuda").int(),
        torch.ones(96*3, 77, device="cuda").bool()
    ]

    with torch.no_grad():
        # You can run this in a new python session!
        model = torch.export.load(f"./models/{out_name}.ep", ).module().cuda()
        model_ = Model().eval().cuda()
        # model = trt_gm
        # model = torch_tensorrt.load("trt.ep").module() # this also works
        out = model(*inputs_)
        out2 = model_(*inputs_)
        out = model(*inputs_)
        out2 = model_(*inputs_)
        import time
        t = time.time()
        out = model(*inputs_)
        print(f"Compiled time: {time.time() - t}")
        import time
        t = time.time()
        out2 = model_(*inputs_)
        print(f"PyTorch time: {time.time() - t}")
        average_error = (torch.abs(out - out2)).mean()
        print("Error:", average_error)
        print()




models = [
    [CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="./models/CLIP").text_model.half().eval().cuda(), "CLIPL4"],
    [AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xxl", cache_dir="./models/T5").encoder.to(torch.float16).eval().cuda(), "T5"],
]


for model, out_name in models:
    compile_model(model, out_name)