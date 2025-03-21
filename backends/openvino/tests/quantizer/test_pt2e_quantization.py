# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

# pyre-unsafe

import unittest
from typing import Dict, Optional, Tuple

import torch
from executorch.backends.openvino.quantizer.quantizer import (
    OpenVINOQuantizer,
    QuantizationMode,
)

from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.torch import disable_patching
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer
from torch.export import export_for_training
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    PT2EQuantizationTestCase,
    TestHelperModules,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    TemporaryFileName,
)


class TestQuantizePT2E(PT2EQuantizationTestCase):

    def run(self, result=None):
        """
        Disable NNCF pathing for each test call
        """
        with disable_patching():
            super().run(result)

    def _get_pt2e_quantized_linear(
        self, mode: Optional[QuantizationMode] = None
    ) -> torch.fx.GraphModule:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        if mode is None:
            quantizer = OpenVINOQuantizer()
        else:
            quantizer = OpenVINOQuantizer(mode=mode)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        return self._quantize(m, quantizer, example_inputs)

    def test_fold_all_ops_before_quantize(self) -> None:
        """Test folding all ops that's before quantized operator:
        Before:
            get_attr(weight) -> transpose -> quantize -> dequantize
        After:
            get_attr(folded_weight) -> dequantize
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(2, 2)

            def forward(self, x):
                t = self.weight.t()
                return torch.nn.functional.linear(x, t)

        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = self._quantize(m, quantizer, example_inputs)
        node_occurrence = {
            # quantize op for weight node is folded
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 1,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 1,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_composable_quantizer_throw(self) -> None:
        class BadQuantizer(Quantizer):
            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for n in gm.graph.nodes:
                    n.meta["quantization_annotation"] = None

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        quantizer = OpenVINOQuantizer()
        bad_quantizer = BadQuantizer()
        composable_quantizer = ComposableQuantizer([quantizer, bad_quantizer])
        m_eager = TestHelperModules.ConvLinearWPermute().eval()
        example_inputs = (torch.randn(2, 3, 4, 4),)
        self.assertRaises(
            RuntimeError,
            lambda: self._test_quantizer(
                m_eager, example_inputs, composable_quantizer, {}
            ),
        )

    @unittest.skip(
        "Enable after the embedding quantization fix: https://github.com/openvinotoolkit/nncf/pull/3237"
    )
    def test_embedding_conv_linear_quantization(self) -> None:
        m_eager = TestHelperModules.EmbeddingConvLinearModule().eval()
        indices = torch.tensor(
            [9, 6, 5, 7, 8, 8, 9, 2, 8, 6]
            + [6, 9, 1, 6, 8, 8, 3, 2, 3, 6]
            + [3, 6, 5, 7, 0, 8, 4, 6, 5, 8]
            + [2, 3]
        )
        indices = torch.unsqueeze(indices, 0)
        example_inputs = (indices,)
        quantizer = OpenVINOQuantizer()

        m = self._quantize(m_eager, quantizer, example_inputs, is_qat=False)

        ref_q = {
            # First conv
            "quantize_per_tensor_default": (
                None,
                0.01585131697356701,
                127,
                0,
                255,
                torch.uint8,
            ),
            "dequantize_per_tensor_default": (
                None,
                0.01585131697356701,
                127,
                0,
                255,
                torch.uint8,
            ),
            "dequantize_per_channel_default": (
                None,
                torch.tensor(
                    [0.0015, 0.0015, 0.0015, 0.0016, 0.0015]
                    + [0.0016, 0.0014, 0.0014, 0.0015, 0.0015]
                    + [0.0016, 0.0015, 0.0015, 0.0016, 0.0016]
                    + [0.0015]
                ),
                torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                0,
                -128,
                127,
                torch.int8,
            ),
            # First linear
            "quantize_per_tensor_default_1": (
                None,
                0.016017982736229897,
                127,
                0,
                255,
                torch.uint8,
            ),
            "dequantize_per_tensor_default_1": (
                None,
                0.016017982736229897,
                127,
                0,
                255,
                torch.uint8,
            ),
            "dequantize_per_channel_default_1": (
                None,
                torch.tensor(
                    [0.0019, 0.0019, 0.0020, 0.0018, 0.0019, 0.0019, 0.0018, 0.0018]
                ),
                torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
                0,
                -128,
                127,
                torch.int8,
            ),
            # TODO: check embedding after the fix in NNCF
        }
        self._check_quantization_with_ref(m, ref_q)

    def test_disallow_eval_train(self) -> None:
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.rand(3, 3, 5, 5),)

        # Before export: this is OK
        m.eval()
        m.train()

        # After export: this is not OK
        m = export_for_training(m, example_inputs).module()
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After prepare: still not OK
        quantizer = OpenVINOQuantizer()
        m = prepare_pt2e(m, quantizer)  # pyre-ignore[6]
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After convert: still not OK
        m = convert_pt2e(m)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

    def _get_bn_train_eval_ops(self) -> Tuple[torch._ops.OpOverload]:
        return (
            torch.ops.aten.batch_norm.default,
            torch.ops.aten.batch_norm.default,
        )

    def _get_node(
        self, m: torch.fx.GraphModule, target: torch._ops.OpOverload
    ) -> torch.fx.Node:
        """
        Return the first node matching the specified target, throwing an exception
        if no such batch norm node is found.
        """
        for n in m.graph.nodes:
            if n.target == target:
                return n
        raise ValueError("Did not find node with target ", target)

    def test_allow_exported_model_train_eval(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.bn(x)
                return x

        m = M().train()
        example_inputs = (torch.randn(1, 3, 3, 3),)
        bn_train_op, bn_eval_op = self._get_bn_train_eval_ops()  # pyre-ignore[23]
        m = export_for_training(m, example_inputs).module()

        def _assert_ops_are_correct(m: torch.fx.GraphModule, train: bool) -> None:
            bn_op = bn_train_op if train else bn_eval_op
            bn_node = self._get_node(m, bn_op)
            self.assertTrue(bn_node is not None)

        # Before wrapping: this is not OK
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After wrapping: does not error and swaps the ops accordingly
        torch.ao.quantization.allow_exported_model_train_eval(m)  # pyre-ignore[6]
        m.eval()
        _assert_ops_are_correct(m, train=False)  # pyre-ignore[6]
        m.train()
        _assert_ops_are_correct(m, train=True)  # pyre-ignore[6]

        # After prepare but before wrapping: this is not OK
        quantizer = OpenVINOQuantizer()
        m = prepare_pt2e(m, quantizer)  # pyre-ignore[6]
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After prepare and after wrapping: does not error and swaps the ops accordingly
        torch.ao.quantization.allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)

        # After convert but before wrapping: this is not OK
        m = convert_pt2e(m, fold_quantize=True)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

        # After convert and after wrapping: does not error and swaps the ops accordingly
        torch.ao.quantization.allow_exported_model_train_eval(m)
        m.eval()
        _assert_ops_are_correct(m, train=False)
        m.train()
        _assert_ops_are_correct(m, train=True)

    def test_constant_prop_preserve_metadata(self) -> None:
        """Test to make sure the get_attr node for const propagated weight Tensor gets the correct
        metadata (from original get_attr node from weight)
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = export_for_training(
            m,
            example_inputs,
        ).module()
        weight_meta = None
        for n in m.graph.nodes:  # pyre-ignore[16]
            if (
                n.op == "get_attr"
                and next(iter(n.users)).target == torch.ops.aten.linear.default
            ):
                weight_meta = n.meta
                break
        assert weight_meta is not None, "Expect to find metadata for weight node"

        m = prepare_pt2e(m, quantizer)  # pyre-ignore[6]
        m(*example_inputs)
        m = convert_pt2e(m)

        for n in m.graph.nodes:
            if n.op == "get_attr" and "frozen_param" in n.target:
                for key in n.meta:
                    self.assertEqual(n.meta[key], weight_meta[key])

    def test_reentrant(self) -> None:
        """Test we can safely call quantization apis multiple times"""
        m = TestHelperModules.ConvBnReLU2dAndLinearReLU()
        example_inputs = (torch.randn(3, 3, 10, 10),)

        quantizer = OpenVINOQuantizer(mode=QuantizationMode.INT8_SYM)
        m.conv_bn_relu = export_for_training(  # pyre-ignore[8]
            m.conv_bn_relu, example_inputs
        ).module()
        m.conv_bn_relu = prepare_pt2e(m.conv_bn_relu, quantizer)  # pyre-ignore[6,8]
        m(*example_inputs)
        m.conv_bn_relu = convert_pt2e(m.conv_bn_relu)  # pyre-ignore[6, 8]

        quantizer = OpenVINOQuantizer(mode=QuantizationMode.INT8_MIXED)
        m = export_for_training(m, example_inputs).module()
        m = prepare_pt2e(m, quantizer)  # pyre-ignore[6]
        m = convert_pt2e(m)

        node_occurrence = {
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 3,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 3,
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_channel.default
            ): 1,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 3,
        }
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_groupwise_per_channel_quant(self) -> None:
        m = TestHelperModules.GroupwiseConv2d()
        quantizer = OpenVINOQuantizer()
        example_inputs = m.example_inputs()
        m = self._quantize(m, quantizer, example_inputs)
        # make sure it runs
        m(*example_inputs)

    def test_preserve_nn_module_stack(self) -> None:
        """Test we can preserve nn_module_stack on replaced pattern's nodes"""
        m = TestHelperModules.ConvBnReLU2dAndLinearReLU()
        example_inputs = (torch.randn(3, 3, 10, 10),)

        quantizer = OpenVINOQuantizer()

        def check_nn_module(node: torch.fx.Node) -> None:
            self.assertTrue("nn_module_stack" in node.meta)
            self.assertTrue(
                "ConvWithBNRelu" in node.meta["nn_module_stack"]["L__self__"][1]
            )

        m.conv_bn_relu = export_for_training(  # pyre-ignore[8]
            m.conv_bn_relu, example_inputs
        ).module()
        for node in m.conv_bn_relu.graph.nodes:  # pyre-ignore[16]
            if node.op not in ["placeholder", "output", "get_attr"]:
                check_nn_module(node)
        m.conv_bn_relu = prepare_pt2e(m.conv_bn_relu, quantizer)  # pyre-ignore[6,8]
        for node in m.conv_bn_relu.graph.nodes:  # pyre-ignore[16]
            if node.name == "mul":
                check_nn_module(node)

    def test_fold_quantize_sym(self) -> None:
        """Test to make sure the quantized model gets quantized weight (quantize_per_tensor op is folded)"""
        m = self._get_pt2e_quantized_linear()

        ref_q = {
            "quantize_per_tensor_default": (
                None,
                0.010390480048954487,
                127,
                0,
                255,
                torch.uint8,
            ),
            "dequantize_per_tensor_default": (
                None,
                0.010390480048954487,
                127,
                0,
                255,
                torch.uint8,
            ),
            "dequantize_per_channel_default": (
                torch.tensor([[-78, -128], [-127, 76]], dtype=torch.int8),
                torch.tensor([0.0029, 0.0036]),
                torch.tensor([0, 0]),
                0,
                -128,
                127,
                torch.int8,
            ),
        }
        self._check_quantization_with_ref(m, ref_q)

    def test_fold_quantize_mixed(self) -> None:
        """Test to make sure the quantized model gets quantized weight (quantize_per_channel op is folded)"""
        m = self._get_pt2e_quantized_linear(mode=QuantizationMode.INT8_MIXED)
        ref_q = {
            "quantize_per_tensor_default": (
                None,
                0.006073841359466314,
                37,
                0,
                255,
                torch.uint8,
            ),
            "dequantize_per_tensor_default": (
                None,
                0.006073841359466314,
                37,
                0,
                255,
                torch.uint8,
            ),
            "dequantize_per_channel_default": (
                torch.tensor([[-78, -128], [-127, 76]], dtype=torch.int8),
                torch.tensor([0.0029, 0.0036]),
                torch.tensor([0, 0]),
                0,
                -128,
                127,
                torch.int8,
            ),
        }
        self._check_quantization_with_ref(m, ref_q)

    def _check_quantization_with_ref(self, model: torch.fx.GraphModule, ref: Dict):
        matches = 0
        for node in model.graph.nodes:
            if node.name not in ref:
                continue
            matches += 1
            ref_values = ref[node.name]
            for idx, ref_value in enumerate(ref_values):
                if ref_value is None:
                    continue
                if isinstance(ref_value, torch.Tensor):
                    self.assertEqual(
                        get_tensor_constant_from_node(node.args[idx], model),
                        ref_value,
                        atol=1e-3,
                        rtol=1e-3,
                    )
                    continue
                if isinstance(ref_value, float):
                    self.assertEqual(node.args[idx], ref_value, atol=1e-3, rtol=1e-3)
                    continue
                assert node.args[idx] == ref_value

        assert len(ref) == matches

    def test_save_load(self) -> None:
        """Test save/load a quantized model"""
        m = self._get_pt2e_quantized_linear()
        example_inputs = (torch.randn(2, 2),)
        ref_res = m(*example_inputs)

        with TemporaryFileName() as fname:
            # serialization
            quantized_ep = torch.export.export(m, example_inputs, strict=True)
            torch.export.save(quantized_ep, fname)
            # deserialization
            loaded_ep = torch.export.load(fname)
            loaded_quantized_model = loaded_ep.module()
            res = loaded_quantized_model(*example_inputs)
            self.assertEqual(ref_res, res)


instantiate_parametrized_tests(TestQuantizePT2E)
