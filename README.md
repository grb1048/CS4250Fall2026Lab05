# Lab 05: The Accelerator You Can't Program

The SG2000 has a neural accelerator on the die: the Cvitek TPU, internally
`bm1822`, 0.5 TOPS of INT8, 32 KB of scratchpad, no instruction set you can
write, no cache, no IOMMU, and no standard. Everything about it is either a
compiler you did not write or a memory map fixed before the kernel booted.

Your job is to put a network of your own on it and then say what it cost you:
the launch overhead, the memory footprint, and the places where the compiler
quietly does the wrong thing.

**There are no models in this directory.** There is a generator that writes six
small ONNX files, a build script, and two input tensors. Every model you run,
you compile — on your laptop, with a toolchain you install yourself. That is
the lab.

Create `REPORT.md` and answer the numbered questions (`(Qx)`) in it. The
questions are the lab; the commands are just how you get to the answers.

Work in pairs and swap pilot/copilot after each part.

## Setup

Your board needs the `-frog` image. It is the Arduino personality plus a
64 MiB ION carveout, and the carveout cannot be added to a running board, so
if you are still on the `-e` image this is a reflash:

```
lz4 -cd duos-e_sd-frog.img.lz4 | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
```

Get the image and `lab05.tar.gz` from <https://files.homework.quest/cs4250/>.
**Back up your home directory before you wipe the card** — that card is where
your project lives. After `dd`, mount the boot partition on your laptop and
copy `lab05.tar.gz` into it, so that on the board you can:

```
tar xzf /boot/lab05.tar.gz          # gives you ~/lab05
cd ~/lab05
```

There is no compiler, no numpy, and no onnxruntime on the board. Everything in
this directory is Python standard library. The toolchain lives on your laptop.

## Check your board

```
sudo cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/total_mem   # 67108864
sudo cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/used_mem    # ~0
lsmod | grep soph_tpu
ls -l /dev/cvi-tpu0
ls /mnt/system/lib
nm -D --defined-only /mnt/system/lib/libcvikernel.so | wc -l      # 313
```

**(Q1)** Five of those six checks pass on the image *without* the carveout —
the module loads, the node exists, the libraries and all 313 symbols are there,
and `used_mem` cheerfully reports zero. Which one is the real test, and why
does `CVI_RT_Init` return success on a board where the first 4 KiB allocation
aborts the process?

## Part 1 — Get a compiler and a model

On your laptop. You need x86-64 Linux or WSL2; on Apple silicon there is no
good path and you should borrow a machine. You also need `git`, `curl` and `uv`
(`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```
git clone https://github.com/NatTuck/sg2000-hints ~/sg2000-hints
~/sg2000-hints/examples/npu/setup-tools.sh     # ~8 minutes, ~3 GB of disk
~/sg2000-hints/examples/npu/smoke/smoke.sh     # prove it works
```

The setup script is short and it handles every sharp edge in that install
(Python 3.10, a yanked OpenCV, `setuptools<80` for `pkg_resources`, a torch
import that happens even for ONNX input, and a bundled glibc that shadows
yours). Run it once and do not think about it again.

Now compile the smallest thing that runs. From your lab directory:

```
REPO=~/sg2000-hints
LAB=~/lab05
cd $LAB

$REPO/tools/venv/bin/python gen_onnx.py onnx/          # six ONNX files, ~20 KB
mkdir -p models

$REPO/tools/venv/bin/model_transform --model_name convnet_32 \
    --model_def onnx/convnet_32.onnx --input_shapes '[[1,3,32,32]]' \
    --mlir convnet_32.mlir

$REPO/tools/venv/bin/model_deploy --mlir convnet_32.mlir --chip cv181x \
    --quantize BF16 --model models/convnet_32_bf16.cvimodel

$REPO/tools/venv/bin/model_tool --info models/convnet_32_bf16.cvimodel
```

Two commands, and the second one is where all the work happens. Read its
output: the `.cvimodel` is not a program. It is a dataflow graph plus a
precomputed command buffer, and `--info` prints the working set the board has
to find.

`./build-models.sh` does the same thing for all seven models this lab uses,
including the 14 MB MobileNetV2 download. Run it now and read what it prints:

```
./build-models.sh
```

It is chatty — progress bars, and a `Can not open hash file` warning that means
nothing. Ignore the noise; read the filename and the ION size at the end of
each step, and the footprint table at the bottom.

## Part 2 — The API is five calls

On the board. `npurun.py` has the `CVI_TENSOR` struct and the library loader
written for you. Fill in the five calls:

```c
CVI_NN_RegisterModel(path, &model);
CVI_NN_GetInputOutputTensors(model, &in, &n_in, &out, &n_out);
memcpy(in[0].sys_mem, data, in[0].mem_size);
CVI_NN_Forward(model, in, n_in, out, n_out);
CVI_NN_CleanupModel(model);
```

scp your `models/` directory across, then:

```
sudo python3 npurun.py models/convnet_32_bf16.cvimodel 100
```

`convnet_32` is 898 parameters and about 1,600 floating-point operations.

**(Q2)** What is the fixed cost of one inference? How do you know that this
model is measuring the launch and not the arithmetic?

**(Q3)** Does the first forward cost more than the rest? What should you
therefore do in every benchmark you write this term?

## Part 3 — Compile a real network

Back on the laptop. `build-models.sh` already downloaded and compiled
MobileNetV2; if you want to see it happen, the two commands are in the script.
It is a 3.5 M-parameter classifier over the 1000 ImageNet classes.

The cvimodel is only worth having if it computes the same thing as the ONNX it
came from. `inputs/dog.npz` is a photo of a yellow labrador, resized to 224x224
and normalised the way ImageNet models expect, and `inputs/imagenet_classes.txt`
is the label list:

```
$REPO/tools/venv/bin/model_runner --input inputs/dog.npz \
    --model models/mobilenetv2_bf16.cvimodel --output dog_out.npz

$REPO/tools/venv/bin/python top5.py dog_out.npz
```

`model_runner` executes the cvimodel on your laptop. **It is a simulator:
never quote a time from it.** For the reference, run the ONNX itself:

```
$REPO/tools/venv/bin/python - <<'EOF'
import numpy as np, onnxruntime as ort
x = np.load("inputs/dog.npz")["data"]
y = ort.InferenceSession("onnx/mobilenetv2-7.onnx",
                         providers=["CPUExecutionProvider"]).run(None, {"data": x})[0]
labels = [l.strip() for l in open("inputs/imagenet_classes.txt")]
for i in np.argsort(y[0])[::-1][:5]:
    print(f"{i:4d}  {labels[i]:32s} {y[0][i]:9.3f}")
EOF
```

You can make your own input from any photo with `prep.py` — it is the
preprocessing the ONNX model does not contain.

**(Q4)** Did your compile produce the same top-1 class as the ONNX model does?
What exactly did you verify, and what did you *not* verify?

Note that `model_deploy` has a `--test_input` flag that is supposed to do this
comparison for you. In our install it dies partway through, in `npz_tool.py`,
on an import error — so the check you just did by hand is the check you have.
That is worth a sentence in your report: a toolchain that cannot validate its
own output is a toolchain you have to test yourself.

## Part 4 — The footprint, and how it dies

On the board. `used_mem` is the carveout. Watch it change as you register a
model.

```
sudo cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/used_mem
sudo python3 npurun.py models/mobilenetv2_bf16.cvimodel 20
sudo cat /sys/kernel/debug/ion/cvi_carveout_heap_dump/used_mem
sudo python3 ion_exhaust.py            # expect a SIGABRT
echo $?
```

**(Q5)** Tabulate model file size against ION footprint for the convnets and
for MobileNetV2. The vendor's own benchmark table quotes 4.40 MB for
mobilenet_v2 and we measure 9.35 MB. What is the difference, and what does it
tell you about the vendor's number?

**(Q6)** How many MobileNetV2 instances fit in the carveout? What happens on
the one after that — and what is the process exit status?

**(Q7)** Why can't `try:`/`except:` save you? What should
`CviDeviceMem::mem_alloc_raw` have returned instead of `assert(0)`, and what
could `CVI_NN_RegisterModel` then have done for its caller?

**(Q8)** Prove from the running kernel that this machine has no IOMMU:

```
zcat /proc/config.gz | grep IOMMU
```

Why does that single fact force the accelerator's memory to be reserved before
the kernel's page allocator exists, and why can't you fix a wrong size by
editing a file on the board?

## Part 5 — The ladder

`models/` has the same 898-parameter convnet compiled at four input sizes:
32x32, 64x64, 128x128 and 256x256. Same weights, same operators; the arithmetic
grows as N² and the launch overhead does not.

```
for n in 32 64 128 256; do
  sudo python3 npurun.py models/convnet_${n}_bf16.cvimodel 200
done
python3 fit.py 32 64 128 256 <your four times>
```

`fit.py` fits `t(N) = F + k * N^2` and reports the fixed cost and the slope.

**(Q9)** Report `F` and `k`. Compare `F` with the number you got in Q2 — do they
agree, and should they? Convert `k` into effective GOPS for this convnet and
compare it with the 0.5 TOPS peak. Where did the other 90-odd percent go?

## Part 6 — Break the compiler

On the laptop, from your lab directory. Three failures, in increasing order of
nastiness. Record the exact message each one gives you.

**FP32.** The chip has no FP32 path at all. Compile MobileNetV2 again, but ask
for F32:

```
$REPO/tools/venv/bin/model_transform --model_name mobilenetv2 \
    --model_def onnx/mobilenetv2-7.onnx --input_shapes '[[1,3,224,224]]' \
    --mlir mobilenetv2.mlir

$REPO/tools/venv/bin/model_deploy --mlir mobilenetv2.mlir --chip cv181x \
    --quantize F32 --model mobilenetv2_f32.cvimodel
```

**(Q10)** What is the error, and at which stage of the pipeline did it happen?
Why is "the model parsed and type-checked, then died when it tried to emit
cv181x operations" a worse experience than a parse error?

**A dynamic shape.** `onnx/dynamic_range.onnx` builds a `Range()` whose length
is computed from the input:

```
$REPO/tools/venv/bin/model_transform --model_name dynamic_range \
    --model_def onnx/dynamic_range.onnx --input_shapes '[[1,8]]' \
    --mlir dynamic_range.mlir
```

**(Q11)** What is the error, what does the message suggest you do about it, and
why doesn't that fix a model whose output length depends on the data? Relate it
to the vocoder in the lecture, which had to be moved to the CPU for exactly
this reason.

**A compiler that fails open.** `onnx/flip_gather.onnx` reverses the four
channels of a 1x4x2x2 tensor using `Gather` with indices `[3,2,1,0]`.

```
$REPO/tools/venv/bin/model_transform --model_name flip_gather \
    --model_def onnx/flip_gather.onnx --input_shapes '[[1,4,2,2]]' \
    --mlir flip_gather.mlir
$REPO/tools/venv/bin/model_deploy --mlir flip_gather.mlir --chip cv181x \
    --quantize BF16 --model flip_gather_bf16.cvimodel
```

Feed it a tensor whose channels are 0, 1, 2, 3 and look at what comes out.
`inputs/flip_in.npz` is that tensor; the answer is four numbers.

**(Q12)** What did you expect and what did you get? How did you find out without
touching the board? Why is a compiler that accepts an operator and silently
ignores it *worse* than one that rejects it — and what test would have caught
this before it shipped?

## Part 7 — Precision

`convnet_32` is compiled three ways: `bf16`, `f16`, and `int8`. Run all three
on the board.

```
sudo python3 npurun.py models/convnet_32_int8.cvimodel 200
sudo python3 npurun.py models/convnet_32_f16.cvimodel 200
$REPO/tools/venv/bin/model_tool --info models/convnet_32_int8.cvimodel   # laptop
```

INT8 needed something the other two did not: a calibration table. That file is
`onnx/convnet_32_cali_table`, produced by running sample images through the
fp32 model and recording each tensor's range. No calibration, no INT8 — the
compiler asserts.

**(Q13)** Compare the times and the footprints of the three. In the tensor map
of the INT8 model, how many `routines` are there, and what is the extra one
doing? Does the extra routine explain the time difference? What would you tell
someone who says "INT8 is the fast mode"?

## Part 8 — What bf16 costs, audibly

`wav/a.wav` and `wav/b.wav` are the same sentence through the same text
encoder. The vocoder that produced them ran twice: once in fp32 on the CPU, once
in bf16 on the TPU. Play them on your laptop. Have your partner tell you which
is which before you look.

```
python3 corr.py wav/a.wav wav/b.wav
```

**(Q14)** Which is the NPU? Could you hear it? The waveforms correlate at about
0.93 — what does that number hide, and which would you ship?

## Part 9 — The crossover

You have `F` and `k` for this accelerator. The lecture has the CPU numbers:
int8 with the vector unit reaches 1.27 GOPS, fp32 reaches 0.29 GFLOP/s, both
hand-written and both unoptimised.

**(Q15)** Predict the N at which an NxN matrix multiply stops being a CPU job
and becomes an accelerator job, using your `F` and the CPU's rate. Compare your
prediction with the measured crossover of about N=128 for int8 and explain the
gap. Then the real question: **which stage of your semester project would you
put on this thing, and what would you have to give up to do it?**

## To Submit

A tarball of your lab directory containing `REPORT.md` and your `npurun.py`.
No models, no `.cvimodel`, no `dog_out.npz`, no binaries.

## If you finish early

- Compile MobileNetV2 with `--quantize INT8`. It needs a calibration set of
  real images (`run_calibration <mlir> --dataset <dir of jpegs> --input_num 100`).
  What does the extra routine cost, and does the top-1 class survive?
- Run 10,000 inferences of MobileNetV2 and plot the time per inference. The
  kernel has a step-wise thermal governor. Does the accelerator throttle?
- `convnet_32_f16.cvimodel` and `convnet_32_bf16.cvimodel` are the same size.
  Which is more accurate, and why would you pick either?
- Take a photo of something on your desk and run it through `prep.py`. Is the
  answer plausible? What does the model do with an image that is not one of the
  1000 classes?

