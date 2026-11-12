#!/usr/bin/env bash
#
# Build every model this lab runs.
#
#   ./build-models.sh
#
# Downloads MobileNetV2, generates the convnets, and compiles all of them for
# cv181x. A few minutes, most of it the 14 MB download and the MobileNetV2
# compile.
#
# The toolchain is chatty -- progress bars, "Can not open hash file" warnings --
# and all of it is harmless. Look for the filename at the end of each step.
#
# Needs the TPU-MLIR toolchain from sg2000-hints/examples/npu/setup-tools.sh.
# Set REPO if your clone is not at ~/sg2000-hints.
#
set -euo pipefail
cd "$(dirname "$0")"

REPO=${REPO:-$HOME/sg2000-hints}
V=$REPO/tools/venv/bin
PY=$V/python
ONNX=onnx
OUT=models
MV2_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"

if [ ! -x "$V/model_deploy" ]; then
    echo "toolchain missing at $V" >&2
    echo "run $REPO/examples/npu/setup-tools.sh first, or set REPO=" >&2
    exit 1
fi

mkdir -p "$ONNX" "$OUT"

# --- the ONNX sources ------------------------------------------------------
# Six files, a few kilobytes each, generated rather than shipped.
echo "== generating the ONNX sources =="
$PY gen_onnx.py "$ONNX"

if [ ! -s "$ONNX/mobilenetv2-7.onnx" ]; then
    echo "== downloading mobilenetv2-7.onnx (14 MB) =="
    curl -fL --progress-bar -o "$ONNX/mobilenetv2-7.onnx" "$MV2_URL"
fi
ls -l "$ONNX/mobilenetv2-7.onnx"

# --- compile ---------------------------------------------------------------
# model_transform: ONNX -> fp32 MLIR, with the input shape nailed down
# model_deploy:    MLIR -> .cvimodel, which is a schedule and not a program
compile() {
    local name=$1 onnx=$2 shape=$3 quant=$4
    shift 4
    local mlir=".build_$name.mlir"
    local out="$OUT/${name}_$(echo "$quant" | tr 'A-Z' 'a-z').cvimodel"
    $V/model_transform --model_name "$name" --model_def "$onnx" \
        --input_shapes "$shape" --mlir "$mlir" >/dev/null
    $V/model_deploy --mlir "$mlir" --chip cv181x --quantize "$quant" \
        --model "$out" "$@" >/dev/null
    rm -f "$mlir"
    echo "  $(basename "$out")"
}

echo "== convnet ladder, bf16 =="
for n in 32 64 128 256; do
    compile "convnet_$n" "$ONNX/convnet_$n.onnx" "[[1,3,$n,$n]]" BF16
done

echo "== convnet_32, the other two precisions =="
compile convnet_32 "$ONNX/convnet_32.onnx" '[[1,3,32,32]]' F16
# INT8 needs a calibration table: the range of every tensor, measured on sample
# inputs. Without one the compiler asserts. The checked-in table came from 20
# random 32x32 images, which is enough to make it compile and not enough to
# make a good model.
compile convnet_32 "$ONNX/convnet_32.onnx" '[[1,3,32,32]]' INT8 \
    --calibration_table "$ONNX/convnet_32_cali_table"

echo "== mobilenetv2, bf16 =="
compile mobilenetv2 "$ONNX/mobilenetv2-7.onnx" '[[1,3,224,224]]' BF16

# --- what did we get -------------------------------------------------------
echo
echo "== footprint (this is the number that decides what fits) =="
for f in "$OUT"/*.cvimodel; do
    ion=$($V/model_tool --info "$f" 2>/dev/null \
        | sed -n 's/.*Need ION Memory Size: (\(.*\))/\1/p')
    printf '  %-34s %10s bytes   %s ION\n' "$(basename "$f")" "$(stat -c%s "$f")" "$ion"
done

cat <<'EOF'

done.  The three failures in Part 6, for reference -- run them yourself:
  F32          -> ASSERT INFO: Not Implemented at TopLowering.h:366
  dynamic      -> UNREACHABLE at Range.cpp:75 (at model_transform, not deploy)
  flip_gather  -> compiles; Gather ignores its indices and returns the input
EOF
