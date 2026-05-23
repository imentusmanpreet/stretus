#!/usr/bin/env bash
# Regenerate Python gRPC stubs from marketdata.proto.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEN="${ROOT}/app/proto/gen"
mkdir -p "${GEN}"
python3 -m grpc_tools.protoc \
  -I"${ROOT}" \
  --python_out="${GEN}" \
  --grpc_python_out="${GEN}" \
  "${ROOT}/marketdata.proto"
# Fix absolute import in generated grpc stub.
sed -i 's/^import marketdata_pb2/from app.proto.gen import marketdata_pb2/' \
  "${GEN}/marketdata_pb2_grpc.py" 2>/dev/null || \
  sed -i '' 's/^import marketdata_pb2/from app.proto.gen import marketdata_pb2/' \
  "${GEN}/marketdata_pb2_grpc.py"
echo "Generated stubs in ${GEN}"
