"""Colour tables that `sv.DepthAnnotator` paints depth maps with.

Each table is 256 sRGB entries as 6-digit hex, darkest or coldest first, in the order
the colour coordinate `t` runs from 0 to 1. They are the published tables matplotlib
ships (`matplotlib.colormaps[name]`, each channel rounded half up to 8 bits; Turbo is
Google's table verbatim) and are byte-identical to supervision-js's
`depth-colormap-tables.ts`, so a map coloured in Python and in the browser looks the
same. `tests/depth/test_colormaps.py` compares them with matplotlib's.
"""

from __future__ import annotations

from enum import Enum
from functools import cache

import numpy as np
import numpy.typing as npt

# Turbo, Anton Mikhailov's published table.
# Copyright 2019 Google LLC.
# SPDX-License-Identifier: Apache-2.0
# https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
_TURBO_TABLE = (
    "30123b32154333184a341b51351e5836215f37246638276d392a733a2d793b2f803c3286"
    "3d358b3e38913f3b973f3e9c4040a24143a74146ac4249b1424bb5434eba4451bf4454c3"
    "4456c74559cb455ccf455ed34661d64664da4666dd4669e0466be3476ee64771e94773eb"
    "4776ee4778f0477bf2467df44680f64682f84685fa4687fb458afc458cfd448ffe4391fe"
    "4294ff4196ff4099ff3e9bfe3d9efe3ba0fd3aa3fc38a5fb37a8fa35abf833adf731aff5"
    "2fb2f42eb4f22cb7f02ab9ee28bceb27bee925c0e723c3e422c5e220c7df1fc9dd1ecbda"
    "1ccdd81bd0d51ad2d21ad4d019d5cd18d7ca18d9c818dbc518ddc218dec018e0bd19e2bb"
    "19e3b91ae4b61ce6b41de7b21fe9af20eaac22ebaa25eca727eea42aefa12cf09e2ff19b"
    "32f29835f39438f4913cf58e3ff68a43f78746f8844af8804ef97d52fa7a55fa7659fb73"
    "5dfc6f61fc6c65fd6969fd666dfe6271fe5f75fe5c79fe597dff5680ff5384ff5188ff4e"
    "8bff4b8fff4992ff4796fe4499fe429cfe409ffd3fa1fd3da4fc3ca7fc3aa9fb39acfb38"
    "affa37b1f936b4f836b7f735b9f635bcf534bef434c1f334c3f134c6f034c8ef34cbed34"
    "cdec34d0ea34d2e935d4e735d7e535d9e436dbe236dde037dfdf37e1dd37e3db38e5d938"
    "e7d739e9d539ebd339ecd13aeecf3aefcd3af1cb3af2c93af4c73af5c53af6c33af7c13a"
    "f8be39f9bc39faba39fbb838fbb637fcb336fcb136fdae35fdac34fea933fea732fea431"
    "fea130fe9e2ffe9b2dfe992cfe962bfe932afe9029fd8d27fd8a26fc8725fc8423fb8122"
    "fb7e21fa7b1ff9781ef9751df8721cf76f1af66c19f56918f46617f36315f26014f15d13"
    "f05b12ef5811ed5510ec530feb500eea4e0de84b0ce7490ce5470be4450ae2430ae14109"
    "df3f08dd3d08dc3b07da3907d83706d63506d43305d23105d02f05ce2d04cc2b04ca2a04"
    "c82803c52603c32503c12302be2102bc2002b91e02b71d02b41b01b21a01af1801ac1701"
    "a91601a71401a41301a112019e10019b0f01980e01950d01920b018e0a018b0902880802"
    "8507028106027e05027a0403"
)

# Viridis, by Stéfan van der Walt, Nathaniel Smith and Eric Firing.
# Released under CC0: https://github.com/BIDS/colormap
_VIRIDIS_TABLE = (
    "44015444025645045745055946075a46085c460a5d460b5e470d60470e61471063471164"
    "47136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173"
    "482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e"
    "46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f85"
    "4240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a"
    "3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c"
    "375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e"
    "31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e"
    "2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e"
    "277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e"
    "23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c"
    "1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f88"
    "1fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa83"
    "25ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b"
    "32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf70"
    "46c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc863"
    "5ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d153"
    "7ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d840"
    "98d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2b"
    "b8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21a"
    "d8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61e"
    "f6e620f8e621fbe723fde725"
)

# Cividis, Nuñez, Anderton and Renslow, PLOS ONE 13(7) e0199239 (2018),
# from https://github.com/pnnl/cmaputil as shipped by matplotlib.
# Copyright (c) 2017, Battelle Memorial Institute. Redistributed under
# its BSD-style licence, which asks for this notice to be kept.
_CIVIDIS_TABLE = (
    "00224e00234f00245100255300255400265600275800285900285b00295d002a5f002a61"
    "002b62002c64002c66002d68002e6a002e6c002f6d00306f003070003170003171013271"
    "0533710833700c34700f357012357014367016377018376f1a386f1c396f1e3a6f203a6f"
    "213b6e233c6e243c6e263d6e273e6e293f6e2a3f6d2b406d2d416d2e416d2f426d31436d"
    "32436d33446d34456c35456c36466c38476c39486c3a486c3b496c3c4a6c3d4a6c3e4b6c"
    "3f4c6c404c6c414d6c424e6c434e6c444f6c45506c46516c47516c48526c49536c4a536c"
    "4b546c4c556c4d556c4e566c4f576c50576c51586d52596d535a6d545a6d555b6d555c6d"
    "565c6d575d6d585e6d595e6e5a5f6e5b606e5c616e5d616e5e626e5e636f5f636f60646f"
    "61656f62656f636670646770656870656870666970676a71686a71696b716a6c716b6d72"
    "6c6d726c6e726d6f726e6f736f7073707173717274727274727374737475747475757575"
    "7676767777767777777878777979777a7a787b7a787c7b787d7c787e7c787e7d787f7e78"
    "807f78817f788280798381798482798582798683798784788885788985788a86788b8778"
    "8c88788d88788e89788f8a78908b78918b78928c78928d78938e78948e77958f77969077"
    "9791779892779992779a93769b94769c95769d95769e96769f9775a09875a19975a29975"
    "a39a74a49b74a59c74a69c74a79d73a89e73a99f73aaa073aba072aca172ada272aea371"
    "afa471b0a571b1a570b3a670b4a76fb5a86fb6a96fb7a96eb8aa6eb9ab6dbaac6dbbad6d"
    "bcae6cbdae6cbeaf6bbfb06bc0b16ac1b26ac2b369c3b369c4b468c5b568c6b667c7b767"
    "c8b866c9b965cbb965ccba64cdbb63cebc63cfbd62d0be62d1bf61d2c060d3c05fd4c15f"
    "d5c25ed6c35dd7c45cd9c55cdac65bdbc75adcc859ddc858dec958dfca57e0cb56e1cc55"
    "e2cd54e4ce53e5cf52e6d051e7d150e8d24fe9d34eead34cebd44bedd54aeed649efd748"
    "f0d846f1d945f2da44f3db42f5dc41f6dd3ff7de3ef8df3cf9e03afbe138fce236fde334"
    "fee434fee535fee636fee838"
)

# Inferno, by Stéfan van der Walt and Nathaniel Smith.
# Released under CC0: https://github.com/BIDS/colormap
_INFERNO_TABLE = (
    "00000401000501010601010802010a02020c02020e030210040312040314050417060419"
    "07051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34"
    "150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51"
    "280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b0964"
    "3d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c"
    "510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e"
    "64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e"
    "771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a"
    "8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b2964"
    "9d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305c"
    "b0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51"
    "c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545"
    "d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337"
    "e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429"
    "eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819"
    "f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0a"
    "f98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309"
    "fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1f"
    "fbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13d"
    "f7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865"
    "f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96"
    "f8fb9af9fc9dfafda1fcffa4"
)

# Magma, by Stéfan van der Walt and Nathaniel Smith.
# Released under CC0: https://github.com/BIDS/colormap
_MAGMA_TABLE = (
    "00000401000501010601010802010902020b02020d03030f030312040414050416060518"
    "06051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31"
    "130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e"
    "22115024125325125527125829115a2a115c2c115f2d11612f1163311165331067341069"
    "36106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f76451077471078491078"
    "4a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f"
    "5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81"
    "701f81721f817320817521817621817822817922827b23827c23827e2482802582812581"
    "8326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80"
    "962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327d"
    "aa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978"
    "bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070"
    "d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67"
    "e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5e"
    "ef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705c"
    "f7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761"
    "fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6c"
    "fe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47b"
    "feb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8d"
    "fecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1"
    "fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8"
    "fcf7b9fcf9bbfcfbbdfcfdbf"
)

_TABLES = {
    "turbo": _TURBO_TABLE,
    "viridis": _VIRIDIS_TABLE,
    "cividis": _CIVIDIS_TABLE,
    "inferno": _INFERNO_TABLE,
    "magma": _MAGMA_TABLE,
}

#: Entries in every depth colour table; the colour coordinate `t` picks one.
DEPTH_COLORMAP_ENTRIES = 256
#: Interpolated steps between two neighbouring table entries in the expanded lookup.
_STEPS_PER_ENTRY = 16
_EXPANDED_ENTRIES = (DEPTH_COLORMAP_ENTRIES - 1) * _STEPS_PER_ENTRY + 1


class DepthColormap(Enum):
    """Colour tables a depth map can be painted with.

    The near end of the depth range is always the warm or bright end of the table.
    Turbo separates the most depth steps and is the default; Viridis and Cividis keep
    their order in grayscale and for colour-blind viewers; Inferno and Magma start
    near black, so pixels without depth should stay unpainted with them.

    Attributes:
        TURBO: Google's Turbo rainbow, built for depth and disparity.
        VIRIDIS: Perceptually uniform, blue to yellow.
        CIVIDIS: Perceptually uniform and optimised for colour-vision deficiency.
        INFERNO: Perceptually uniform, black to pale yellow.
        MAGMA: Perceptually uniform, black to pale pink.
        GRAYSCALE: Black far, white near.
    """

    TURBO = "turbo"
    VIRIDIS = "viridis"
    CIVIDIS = "cividis"
    INFERNO = "inferno"
    MAGMA = "magma"
    GRAYSCALE = "grayscale"

    @classmethod
    def list(cls) -> list[str]:
        """Return the string value of every colormap."""
        return [member.value for member in cls]

    @classmethod
    def from_value(cls, value: DepthColormap | str) -> DepthColormap:
        """Resolve a colormap from an enum member or its case-insensitive name.

        Args:
            value: A `DepthColormap` member or one of its string values.

        Returns:
            The matching `DepthColormap`.

        Raises:
            ValueError: If `value` names no colormap.

        Examples:
            ```pycon
            >>> import supervision as sv
            >>> sv.DepthColormap.from_value("Viridis")
            <DepthColormap.VIRIDIS: 'viridis'>

            ```
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        raise ValueError(
            f"Invalid depth colormap: {value!r}. Must be one of {cls.list()}."
        )

    def rgb_lut(self) -> npt.NDArray[np.uint8]:
        """Return the colour table as a `(256, 3)` RGB array, far end first.

        Entry `i` is the colour of the colour coordinate `t = i / 255`. The table is
        the one `sv.DepthAnnotator` paints with, so it can build a colour bar that
        matches the picture, for example with
        `matplotlib.colors.ListedColormap(lut / 255)`.

        Returns:
            A new `(256, 3)` `uint8` array in RGB order.

        Examples:
            ```pycon
            >>> import supervision as sv
            >>> lut = sv.DepthColormap.TURBO.rgb_lut()
            >>> lut.shape
            (256, 3)
            >>> lut[0].tolist(), lut[-1].tolist()
            ([48, 18, 59], [122, 4, 3])

            ```
        """
        lut: npt.NDArray[np.uint8] = _rgb_lut(self).copy()
        return lut


@cache
def _rgb_lut(colormap: DepthColormap) -> npt.NDArray[np.uint8]:
    """Decode a colormap's hex table once into a read-only `(256, 3)` RGB array."""
    lut: npt.NDArray[np.uint8]
    if colormap is DepthColormap.GRAYSCALE:
        ramp = np.arange(DEPTH_COLORMAP_ENTRIES, dtype=np.uint8)
        lut = np.repeat(ramp[:, np.newaxis], 3, axis=1)
    else:
        raw = bytes.fromhex(_TABLES[colormap.value])
        lut = np.frombuffer(raw, dtype=np.uint8).reshape(DEPTH_COLORMAP_ENTRIES, 3)
    lut = lut.copy()
    lut.flags.writeable = False
    return lut


@cache
def _expanded_bgr_lut(colormap: DepthColormap) -> npt.NDArray[np.uint8]:
    """Return the table interpolated to 16 steps per entry, in BGR order.

    supervision-js samples its 256x1 table texture with linear filtering at
    `u = (t * 255 + 0.5) / 256`, so a colour between two entries is their linear
    blend. Expanding the table once to `255 * 16 + 1` entries lets one `np.take` per
    frame reproduce that blend to a sixteenth of an entry; entry `16 * i` is exactly
    table entry `i`.
    """
    rgb = _rgb_lut(colormap).astype(np.float64)
    positions = np.arange(_EXPANDED_ENTRIES) / _STEPS_PER_ENTRY
    expanded = np.column_stack(
        [
            np.interp(positions, np.arange(DEPTH_COLORMAP_ENTRIES), rgb[:, channel])
            for channel in range(3)
        ]
    )
    bgr: npt.NDArray[np.uint8] = np.rint(expanded[:, ::-1]).astype(np.uint8)
    bgr.flags.writeable = False
    return bgr


def _colorize(
    t: npt.NDArray[np.floating], colormap: DepthColormap
) -> npt.NDArray[np.uint8]:
    """Map colour coordinates in `[0, 1]` to BGR colours, shape `t.shape + (3,)`."""
    index = np.rint(t * (_EXPANDED_ENTRIES - 1)).astype(np.intp)
    colors: npt.NDArray[np.uint8] = np.take(_expanded_bgr_lut(colormap), index, axis=0)
    return colors
