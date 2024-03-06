import itertools
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair, _single

from .orthogonal import Orthogonal, cayley_init_, cayley_map


class CSP(Orthogonal):
    def __init__(
        self,
        num_channels: int,
        num_features: int = None,
        # stride: _size_2_t = 1,
        # padding: Union[str, _size_2_t] = 0,
        # dilation: _size_2_t = 1,
        # groups: int = 1,
        # bias: bool = True,
        # padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_channels = num_channels
        self.num_features = num_channels if num_features is None else num_features
        super(CSP, self).__init__(
            self.num_channels, self.num_features, cayley_init_, "static", cayley_map
        )

    def get_weights(self) -> Tensor:
        weight = self.B()
        weight = torch.permute(weight, (1, 0))
        weight = weight.unsqueeze(2).unsqueeze(1)
        return weight

    def _conv_forward(self, input: Tensor):
        weight = self.get_weights()
        return F.conv2d(input, weight, None, _single(1), _single(0), _single(1), 1)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input)


class CSPNN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_features: int = None,
        num_bands: int = None,
        num_windows: int = 1,
        num_labels: int = None,
        csp_pow: bool = True,
        mode: str = "constant",
    ):
        assert (
            num_bands is not None
        ), f"expected 'num_bands' to be int but got {num_bands}"
        assert (
            num_labels is not None
        ), f"expected 'num_labels' to be int but got {num_labels}"

        super(CSPNN, self).__init__()
        self.num_channels = num_channels
        self.num_features = num_channels if num_features is None else num_features
        self.num_bands = num_bands
        self.num_windows = num_windows
        self.num_labels = num_labels
        self.csp_pow = csp_pow
        self.mode = mode

        self.template = "label-{}_band-{}_window-{}"

        self.cspw = nn.ModuleDict(
            [
                [
                    self.template.format(*i),
                    nn.Conv2d(
                        1,
                        self.num_features,
                        (self.num_channels, 1),
                        padding=0,
                        bias=False,
                    )
                    if self.mode == "constant"
                    else CSP(self.num_channels, self.num_features),
                ]
                for i in itertools.product(
                    range(1, self.num_labels + 1),
                    range(self.num_bands),
                    range(self.num_windows),
                )
            ]
        )

    def _csp_post_projection(self, x):
        # [filters x signals] . [signals x filters] = [filters x filters]
        x = torch.matmul(x, x.transpose(3, 2))  # [batch, band, filters, filters]
        # print(x.size())

        num = torch.diagonal(x, dim1=2, dim2=3)  # [batch, band, filters]
        # print(f"{num.size()=}")
        den = torch.sum(num, dim=2).unsqueeze(2)  # [batch, band, 1]
        # print(f"{den.size()=}")

        csp = torch.log(torch.div(num, den))  # [batch, band, filters]
        return csp

    def forward(self, x):
        # [batch, channel, window, band, signal] to [batch, window, band, channel, signal]
        x = x.permute(0, 2, 3, 1, 4)

        x = torch.vstack(
            [
                self.cspw[self.template.format(*i)](
                    x[:, i[2], i[1], :, :].unsqueeze(1)
                ).unsqueeze(0)
                for i in itertools.product(
                    range(1, self.num_labels + 1),
                    range(self.num_bands),
                    range(self.num_windows),
                )
            ]
        ).squeeze()  # [batch, band * window * label, filters(kernels), signal]
        # print(f"{x.size() = }")

        if len(x.size()) <= 3:
            x = x.unsqueeze(0)
        else:
            x = x.permute(1, 0, 2, 3)
            # print(f"{x.size() = }")

        if self.csp_pow:
            csp = self._csp_post_projection(x)
            # print(f"{csp.size()=}")
        else:
            csp = x
            # print(f"{csp.size()=}")

        return csp

    def load_csp_weights(self, weights, csp_template=None, freeze: bool = True):
        if csp_template is None:
            csp_template = self.template

        for i in itertools.product(
            range(1, self.num_labels + 1),
            range(self.num_bands),
            range(self.num_windows),
        ):
            dtype = self.cspw[self.template.format(*i)].weight.dtype
            assert (
                self.cspw[self.template.format(*i)].weight.squeeze().size()
                == weights[csp_template.format(*i)].shape
            )
            with torch.no_grad():
                self.cspw[self.template.format(*i)].weight = nn.Parameter(
                    torch.tensor(weights[csp_template.format(*i)], dtype=dtype)
                    .T.unsqueeze(1)
                    .unsqueeze(3)
                )
            if freeze:
                for param in self.cspw[self.template.format(*i)].parameters():
                    param.requires_grad = False
