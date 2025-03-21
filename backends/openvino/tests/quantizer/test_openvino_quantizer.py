# Owner(s): ["oncall: mobile"]
import copy
import unittest

import torch
import torch._dynamo as torchdynamo
from executorch.backends.openvino.quantizer.quantizer import OpenVINOQuantizer

from nncf.torch import disable_patching
from torch.ao.quantization import default_dynamic_qconfig

from torch.export import export_for_training
from torch.testing._internal.common_quantization import (
    PT2EQuantizationTestCase,
    TestHelperModules,
)


class TestOpenVINOQuantizer(PT2EQuantizationTestCase):
    def run(self, result=None):
        """
        Disable NNCF pathing for each test call
        """
        with disable_patching():
            super().run(result)

    def test_conv1d(self):
        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(1, 3, 5),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv1d.default,
        ]
        self._test_quantizer(
            TestHelperModules.ConvWithBNRelu(dim=1, relu=False, bn=False),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_conv2d(self):
        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
        ]
        self._test_quantizer(
            TestHelperModules.ConvWithBNRelu(relu=False, bn=False),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_conv1d_with_conv2d(self):
        quantizer = OpenVINOQuantizer()
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv1d.default,
        ]
        m = TestHelperModules.Conv2dThenConv1d()
        self._test_quantizer(
            m,
            m.example_inputs(),
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_linear(self):
        quantizer = OpenVINOQuantizer()
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # Test with 2d inputs
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_3d = (torch.randn(9, 10, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        for example_inputs in [example_inputs_2d, example_inputs_3d, example_inputs_4d]:
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],
            )

    def test_linear_relu(self):
        quantizer = OpenVINOQuantizer()
        m_eager = TestHelperModules.LinearReluModel().eval()

        # Test with 2d inputs
        example_inputs_2d = (torch.randn(1, 5),)
        example_inputs_3d = (torch.randn(1, 2, 5),)
        example_inputs_4d = (torch.randn(1, 2, 3, 5),)

        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        for example_inputs in [example_inputs_2d, example_inputs_3d, example_inputs_4d]:
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],  # node_list
                False,  # executorch_backend_config() does not fuse linear-relu
            )

    def test_conv_linear_no_permute(self):
        quantizer = OpenVINOQuantizer()
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
        }
        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        self._test_quantizer(
            TestHelperModules.Conv2dWithTwoLinear(),
            example_inputs,
            quantizer,
            node_occurrence,
            [],
        )

    def test_conv_linear(self):
        quantizer = OpenVINOQuantizer()

        # Test with 2d inputs
        example_inputs = (torch.randn(2, 3, 4, 4),)
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.permute.default,
        ]
        self._test_quantizer(
            TestHelperModules.Conv2dWithTwoLinearPermute(),
            example_inputs,
            quantizer,
            node_occurrence,
            expected_node_list=node_list,
        )

    def test_linear_with_dynamic_shape(self):
        quantizer = OpenVINOQuantizer()
        m_eager = TestHelperModules.TwoLinearModule().eval()

        # Test with 2d inputs
        example_inputs_3d = (torch.randn(9, 10, 8),)
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        self._test_quantizer(
            m_eager,
            example_inputs_3d,
            quantizer,
            node_occurrence,
            [],
            export_with_dynamic_shape=True,
        )

    def test_obs_sharing_ops(self):
        quantizer = OpenVINOQuantizer()
        m = TestHelperModules.Conv2dWithObsSharingOps().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # quantize_per_channel for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.adaptive_avg_pool2d.default,
            torch.ops.aten.hardtanh.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.mean.default,
        ]
        self._test_quantizer(m, example_inputs, quantizer, node_occurrence, node_list)

    def test_propagate_annotation(self):
        quantizer = OpenVINOQuantizer()
        m = TestHelperModules.Conv2dPropAnnotaton().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.view.default,
            torch.ops.aten.hardtanh.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
        ]
        self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            expected_node_occurrence=node_occurrence,
            expected_node_list=node_list,
        )

    def test_dynamic_linear(self):
        quantizer = OpenVINOQuantizer()
        m_eager = TestHelperModules.TwoLinearModule().eval()

        node_occurrence = {
            # input and output are using quantize_per_tensor and weight is using quantize_per_channel
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        # Test with 2d inputs
        example_inputs_2d = (torch.randn(9, 8),)
        example_inputs_4d = (torch.randn(9, 10, 11, 8),)
        for example_inputs in [example_inputs_2d, example_inputs_4d]:
            self._test_quantizer(
                m_eager,
                example_inputs,
                quantizer,
                node_occurrence,
                [],
            )

    @unittest.skip("gru quantization is not supported yet by OpenVINOQuantizer")
    def test_gru(self):
        """this is a test for annotating fp32 GRU so that it produces
        q -> dq -> fp32_gru -> q -> dq, this is currently enough for our use cases,
        but we may change the annotation to be more precise in the future
        """

        class RNNDynamicModel(torch.nn.Module):
            def __init__(self, mod_type):
                super().__init__()
                self.qconfig = default_dynamic_qconfig
                if mod_type == "GRU":
                    self.mod = torch.nn.GRU(2, 2).to(dtype=torch.float)
                if mod_type == "LSTM":
                    self.mod = torch.nn.LSTM(2, 2).to(dtype=torch.float)

            def forward(self, input_tensor, hidden_tensor):
                input_tensor = 1 * input_tensor
                hidden_tensor = 1 * hidden_tensor
                output_tensor, hidden_out = self.mod(input_tensor, hidden_tensor)
                return 1 * output_tensor, 1 * hidden_out

        model_fx = RNNDynamicModel("GRU")
        niter = 10
        example_inputs = (
            # input_tensor
            torch.tensor([[100, -155], [-155, 100], [100, -155]], dtype=torch.float)
            .unsqueeze(0)
            .repeat(niter, 1, 1),
            # hidden_tensor
            # (D * num_layers, N, H_out)
            torch.tensor([[[100, -155]]], dtype=torch.float).repeat(1, 3, 1),
        )
        model_graph = copy.deepcopy(model_fx)

        with torchdynamo.config.patch(allow_rnn=True):
            model_graph = export_for_training(
                model_graph,
                example_inputs,
            ).module()
        quantizer = OpenVINOQuantizer()

        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 0,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 0,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
        }

        with torchdynamo.config.patch(allow_rnn=True):
            self._test_quantizer(
                model_graph,
                example_inputs,
                quantizer,
                expected_node_occurrence=node_occurrence,
            )

    @unittest.skip("gru quantization is not supported yet by OpenVINOQuantizer")
    def test_linear_gru(self):
        """this test is to make sure GRU annotation does not interfere with linear annotation"""

        class RNNDynamicModel(torch.nn.Module):
            def __init__(self, mod_type):
                super().__init__()
                self.qconfig = default_dynamic_qconfig
                self.linear = torch.nn.Linear(2, 2)
                if mod_type == "GRU":
                    self.mod = torch.nn.GRU(2, 2).to(dtype=torch.float)
                if mod_type == "LSTM":
                    self.mod = torch.nn.LSTM(2, 2).to(dtype=torch.float)

            def forward(self, input_tensor, hidden_tensor):
                input_tensor = self.linear(input_tensor)
                input_tensor = 1 * input_tensor
                hidden_tensor = 1 * hidden_tensor
                output_tensor, hidden_out = self.mod(input_tensor, hidden_tensor)
                return 1 * output_tensor, 1 * hidden_out

        model_fx = RNNDynamicModel("GRU")
        niter = 10
        example_inputs = (
            # input_tensor
            torch.tensor([[100, -155], [-155, 100], [100, -155]], dtype=torch.float)
            .unsqueeze(0)
            .repeat(niter, 1, 1),
            # hidden_tensor
            # (D * num_layers, N, H_out)
            torch.tensor([[[100, -155]]], dtype=torch.float).repeat(1, 3, 1),
        )
        model_graph = copy.deepcopy(model_fx)
        quantizer = OpenVINOQuantizer()
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 0,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 0,
            # note: quantize op for weights are const propagated
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
        }
        with torchdynamo.config.patch(allow_rnn=True):
            self._test_quantizer(
                model_graph,
                example_inputs,
                quantizer,
                expected_node_occurrence=node_occurrence,
            )

    def test_add_and_inplace_add(self):
        quantizer = OpenVINOQuantizer()
        example_inputs = (
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
        )
        node_occurrence = {
            # two input and one output for first add, and output for second add
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.add.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        self._test_quantizer(
            TestHelperModules.AddInplaceAdd(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_mul_and_inplace_mul(self):
        quantizer = OpenVINOQuantizer()
        example_inputs = (
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
        )
        node_occurrence = {
            # two input and one output for first add, and output for second add
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            # TODO torch.ops.aten.mul.Tensor,
        ]
        self._test_quantizer(
            TestHelperModules.MulInplaceMul(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_add_mul_scalar(self):
        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 0,
        }

        self._test_quantizer(
            TestHelperModules.AddMulScalar(),
            example_inputs,
            quantizer,
            node_occurrence,
        )

    def test_mul_float32_max(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x * 3.4028235e38

        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # not quantized
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 0,
        }
        node_list = [
            torch.ops.aten.mul.Tensor,
        ]

        self._test_quantizer(
            M(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_add_mul_long(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.t = torch.tensor([100])

            def forward(self, x):
                x = x + self.t
                x = x * self.t
                return x

        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        # not quantized
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 0,
        }
        node_list = [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]
        self._test_quantizer(
            M(),
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    def test_ignored_scope(self):
        for kwargs in [
            {"types": ["linear"]},
            {"names": ["linear", "linear_1"]},
            {"patterns": ["linear*"]},
        ]:
            quantizer = OpenVINOQuantizer()
            quantizer.set_ignored_scope(**kwargs)

            # Test with 2d inputs
            example_inputs = (torch.randn(2, 3, 4, 4),)
            node_occurrence = {
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                # quantize_per_channel for weights are const propagated
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
            }
            node_list = [
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.permute.default,
            ]
            self._test_quantizer(
                TestHelperModules.Conv2dWithTwoLinearPermute(),
                example_inputs,
                quantizer,
                node_occurrence,
                expected_node_list=node_list,
            )
