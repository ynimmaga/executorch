# Owner(s): ["oncall: quantization"]
import copy
import unittest
from typing import Any, Optional

import torch
from executorch.backends.openvino.quantizer.quantizer import OpenVINOQuantizer

from nncf.torch import disable_patching
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import Quantizer
from torch.export import export_for_training
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    TestHelperModules,
)


class TestPT2ERepresentation(QuantizationTestCase):
    def run(self, result=None):
        """
        Disable NNCF pathing for each test call
        """
        with disable_patching():
            super().run(result)

    def _test_representation(
        self,
        model: torch.nn.Module,
        example_inputs: tuple[Any, ...],
        quantizer: Quantizer,
        ref_node_occurrence: dict[ns, int],
        non_ref_node_occurrence: dict[ns, int],
        fixed_output_tol: Optional[float] = None,
        output_scale_idx: int = 1,
    ) -> None:
        # resetting dynamo cache
        torch._dynamo.reset()
        model = export_for_training(
            model,
            example_inputs,
        ).module()
        model_copy = copy.deepcopy(model)

        model = prepare_pt2e(model, quantizer)  # pyre-ignore[6]
        # Calibrate
        model(*example_inputs)
        model = convert_pt2e(model, use_reference_representation=True)
        self.checkGraphModuleNodes(model, expected_node_occurrence=ref_node_occurrence)
        # make sure it runs
        pt2e_quant_output = model(*example_inputs)

        # TODO: torchdynamo times out when we do this, we can enable numerical checking
        # after that is fixed
        model_copy = prepare_pt2e(model_copy, quantizer)  # pyre-ignore[6]
        # Calibrate
        model_copy(*example_inputs)
        model_copy = convert_pt2e(model_copy, use_reference_representation=False)
        self.checkGraphModuleNodes(
            model_copy, expected_node_occurrence=non_ref_node_occurrence
        )
        pt2e_quant_output_copy = model_copy(*example_inputs)

        output_tol = None
        if fixed_output_tol is not None:
            output_tol = fixed_output_tol
        else:
            idx = 0
            for n in model_copy.graph.nodes:
                if (
                    n.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                ):
                    idx += 1
                    if idx == output_scale_idx:
                        output_tol = n.args[1]
                        break
            assert output_tol is not None

        # make sure the result is off by one at most in the quantized integer representation
        self.assertTrue(
            torch.max(torch.abs(pt2e_quant_output_copy - pt2e_quant_output))
            <= (2 * output_tol + 1e-5)
        )

    def test_static_linear(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(2, 5),)

        self._test_representation(
            M().eval(),
            example_inputs,
            quantizer,
            ref_node_occurrence={},
            non_ref_node_occurrence={},
        )

    def test_dynamic_linear(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x)

        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(2, 5),)

        self._test_representation(
            M().eval(),
            example_inputs,
            quantizer,
            ref_node_occurrence={},
            non_ref_node_occurrence={},
            fixed_output_tol=1e-4,
        )

    def test_conv2d(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2d = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv2d(x)

        quantizer = OpenVINOQuantizer()
        example_inputs = (torch.randn(1, 3, 3, 3),)

        self._test_representation(
            M().eval(),
            example_inputs,
            quantizer,
            ref_node_occurrence={},
            non_ref_node_occurrence={},
        )

    def test_maxpool2d(self):
        quantizer = OpenVINOQuantizer()
        m_eager = TestHelperModules.ConvMaxPool2d().eval()

        example_inputs = (torch.randn(1, 2, 2, 2),)

        self._test_representation(
            m_eager,
            example_inputs,
            quantizer,
            ref_node_occurrence={},
            non_ref_node_occurrence={},
        )
