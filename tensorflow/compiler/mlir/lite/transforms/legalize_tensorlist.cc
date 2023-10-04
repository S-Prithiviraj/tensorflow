/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

mlir::TFL::ConstBytesAttr SerializeOptionsToBytes(
    mlir::MLIRContext* context, const std::vector<uint8_t>& options) {
  std::string content;
  content.assign(reinterpret_cast<const char*>(options.data()), options.size());
  return mlir::TFL::ConstBytesAttr::get(context, content);
}

mlir::TFL::ConstBytesAttr CreateListReserveOptions(
    mlir::MLIRContext* context, tflite::TensorType element_type) {
  std::vector<uint8_t> options;
  options.push_back(element_type);
  return SerializeOptionsToBytes(context, options);
}

std::optional<mlir::Type> GetSingularVariantBaseType(mlir::Value val) {
  auto val_t =
      mlir::getElementTypeOrSelf(val).dyn_cast_or_null<mlir::TF::VariantType>();
  if (!val_t) {
    return std::nullopt;
  }
  llvm::ArrayRef<mlir::TensorType> subtypes = val_t.getSubtypes();
  if (subtypes.size() != 1) {
    return std::nullopt;
  }
  return subtypes[0].getElementType();
}  // NOLINT: TODO(b/257472333) This function will be used in child changes.

}  // namespace

// Create an `mlir::TFL::ConstBytesAttr` which encodes the options
// for the `tf.custom` tensor list op to be created. If the given
// op is not a `tf.TensorList*` op, return empty, although this case
// should never be trigged in practice since patterns are only applied
// on `tf.TensorList*` ops.
std::optional<mlir::TFL::ConstBytesAttr> CustomOptions(
    mlir::MLIRContext* context, mlir::Operation* op) {
  if (auto reserve =
          llvm::dyn_cast_or_null<mlir::TF::TensorListReserveOp>(op)) {
    tflite::TensorType tflite_type =
        tflite::ConvertTypeToTensorType(reserve.getElementDtype());

    return CreateListReserveOptions(context, tflite_type);
  }
  if (auto empty = llvm::dyn_cast_or_null<mlir::TF::EmptyTensorListOp>(op)) {
    tflite::TensorType tflite_type =
        tflite::ConvertTypeToTensorType(empty.getElementDtype());

    return CreateListReserveOptions(context, tflite_type);
  }
  return {};
}

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_LEGALIZETENSORLISTPASS
#include "tensorflow/compiler/mlir/lite/transforms/generated_legalize_tensorlist.inc"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

bool IsOpSupported(mlir::Operation* op) {
  // Op is vacuously "supported" if it is not a tensorlist op.
  StringRef op_name = op->getName().getStringRef();
  if (!op_name.contains("TensorList")) return true;

  std::optional<mlir::Type> element_type = {};

  if (auto reserve = llvm::dyn_cast_or_null<TF::TensorListReserveOp>(op)) {
    element_type = reserve.getElementDtype();
  }
  if (auto stack = llvm::dyn_cast_or_null<TF::TensorListStackOp>(op)) {
    element_type = stack.getElementDtype();
  }
  if (auto set_item = llvm::dyn_cast_or_null<TF::TensorListSetItemOp>(op)) {
    element_type = set_item.getElementDtype();
  }
  if (auto from_tensor =
          llvm::dyn_cast_or_null<TF::TensorListFromTensorOp>(op)) {
    element_type = from_tensor.getElementDtype();
  }
  if (auto get_item = llvm::dyn_cast_or_null<TF::TensorListGetItemOp>(op)) {
    element_type = get_item.getElementDtype();
  }
  if (auto empty = llvm::dyn_cast_or_null<TF::EmptyTensorListOp>(op)) {
    element_type = empty.getElementDtype();
  }
  if (auto element_shape =
          llvm::dyn_cast_or_null<TF::TensorListElementShapeOp>(op)) {
    element_type = GetSingularVariantBaseType(op->getOperand(0));
  }

  if (!element_type.has_value()) return false;
  // TODO(b/288302706) add support for all types handled in the
  // `lower_static_tensor_list` pass.
  return element_type->isF32() || element_type->isInteger(64) ||
         element_type->isInteger(32) || element_type->isInteger(1);
}

bool HasVariantInputOrOutput(Operation* op) {
  const bool has_variant_input = llvm::any_of(op->getOperands(), [](Value val) {
    return val.getType()
        .cast<TensorType>()
        .getElementType()
        .isa<TF::VariantType>();
  });
  const bool has_variant_output =
      llvm::any_of(op->getResultTypes(), [](Type t) {
        return t.cast<TensorType>().getElementType().isa<TF::VariantType>();
      });
  return has_variant_input || has_variant_output;
}

// There are 2 standard tf ops which are not TensorList ops that may take as
// input a tensorlist. These are tf.AddN and tf.ZeroesLike. Since the runtime
// implementation of a tensorlist are not compatible between tf and tflite
// we cannot use tflite tensorlist kernels until these cases are handled.
bool IsNonTensorListVariantOp(Operation* op) {
  return llvm::isa<TF::AddNOp, TF::ZerosLikeOp>(op) &&
         HasVariantInputOrOutput(op);
}

// Only legalize TensorFlow TensorList ops if all TensorList ops are supported
// natively.
class LegalizeTensorListPass
    : public impl::LegalizeTensorListPassBase<LegalizeTensorListPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeTensorListPass)

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    auto walk_res = module->walk([&](Operation* op) -> WalkResult {
      if (!IsOpSupported(op) || IsNonTensorListVariantOp(op)) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (walk_res.wasInterrupted()) {
      llvm::errs() << "Tried legalizing to tfl custom tensorlist ops, but not "
                      "all can be supported."
                   << "\n";
      return;
    }

    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTensorListPass() {
  return std::make_unique<LegalizeTensorListPass>();
}

}  // namespace TFL
}  // namespace mlir
