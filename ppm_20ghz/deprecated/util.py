import numpy as np
import orjson
from load_schema import GMDataPhotonCommunity, GMData
import copy
from pydantic import BaseModel
import networkx as nx
from kl_divergence import kl_mvn
import math
import matplotlib.pyplot as plt
import load_schema

import colorsys


def gaussian_2d(x, y, cov, pos=(0, 0)):
    """Generate a 2D Gaussian distribution with a given covariance matrix and position."""
    # Unpack the position tuple
    x0, y0 = pos

    # Shift the coordinates by x0 and y0
    x = x - x0
    y = y - y0

    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)

    # Calculate the determinant of the covariance matrix
    det_cov = np.linalg.det(cov)

    # Calculate the exponent term of the Gaussian function
    exponent = -0.5 * (
        inv_cov[0, 0] * x**2
        + (inv_cov[0, 1] + inv_cov[1, 0]) * x * y
        + inv_cov[1, 1] * y**2
    )

    # Calculate the normalization constant of the Gaussian function
    norm_const = 1 / (2 * np.pi * np.sqrt(det_cov))

    # Calculate the Gaussian function values
    gauss = norm_const * np.exp(exponent)

    return gauss


class PhotonGMData(BaseModel):
    center_of_mass: list[float]
    gm_photon: GMDataPhotonCommunity


def remove_far_field_gaussians(
    gm_data: GMData, gaussian_distance_cutoff: float = 500
) -> GMData:
    # print("gm_data.weights: ", gm_data.weights[:5])
    # print("sum of weights: ", np.sum(gm_data.weights))

    gm_center = np.average(gm_data.means, weights=gm_data.weights, axis=0)
    print(gm_center)

    # Create a boolean mask for the elements to keep
    mask = np.linalg.norm(gm_data.means - gm_center, axis=1) <= gaussian_distance_cutoff

    # Apply the mask to the arrays
    gm_data.covariances = gm_data.covariances[mask]
    gm_data.means = gm_data.means[mask]
    gm_data.weights = gm_data.weights[mask]

    # Recompute the weights so that their sum is 1
    gm_data.weights = gm_data.weights / np.sum(gm_data.weights)

    return gm_data


def filter_unwanted_gaussians(
    gm_data: GMData,
    stretch_ratio_cutoff: float = 2.0,
    large_cutoff: float = 1500,
    small_cutoff: float = 10,
    weight_cutoff: float = 0.005,
) -> GMData:
    # Calculate the SVD of each covariance matrix
    singular_values = np.linalg.svd(gm_data.covariances, compute_uv=False)

    # Calculate the ratio of the singular values
    ratios = singular_values[:, 0] / singular_values[:, 1]
    weights = gm_data.weights

    # Calculate the sum of the singular values
    sigmas = np.sum(singular_values, axis=1)

    # Create a boolean mask for the elements to keep
    mask = (
        (ratios <= stretch_ratio_cutoff)
        & (sigmas <= large_cutoff)
        & (sigmas >= small_cutoff)
        & (weights > weight_cutoff)
    )

    # Apply the mask to the arrays
    gm_data.covariances = gm_data.covariances[mask]
    gm_data.means = gm_data.means[mask]
    gm_data.weights = gm_data.weights[mask]

    # Recompute the weights so that their sum is 1
    gm_data.weights = gm_data.weights / np.sum(gm_data.weights)

    return gm_data


def photon_number_bifurcate_gm_data(
    gm_data: GMData, res: list[list[int]]
) -> list[PhotonGMData]:
    """seperate out the gm_data gaussians into groups for each photon number, based on the sets of gaussians specified in res

    Args:
        gm_data (GMData): _description_
        res (list[list[int]]): _description_
        gaussian_distance_cutoff (float, optional): _description_. Defaults to 500.

    Returns:
        list[PhotonGMData]:
    """
    gm_center = np.average(gm_data.means, weights=gm_data.weights, axis=0)
    photon_gm_datas = []
    for st in res:
        covariances = []
        means = []
        weights = []
        for item in st:
            covariances.append(gm_data.covariances[item])
            means.append(gm_data.means[item])
            weights.append(gm_data.weights[item])

        # normalize the community as a whole to 1
        weights = np.array(weights) / np.sum(weights)
        weights = weights.tolist()

        # log likelihood is not valid here
        photon_number_gm_data = GMDataPhotonCommunity(
            photon_covariances=covariances,
            photon_means=means,
            photon_weights=weights,
            photon_num_components=len(st),
        )
        center_of_mass = np.average(means, weights=weights, axis=0)
        pgm = PhotonGMData(
            center_of_mass=center_of_mass, gm_photon=photon_number_gm_data
        )

        photon_gm_datas.append(pgm)

    # now we have a list of photon number gaussian mixture groups.
    # But it's not clear which photon number they correspond to.
    # Luckily this can be inferred from their center of mass

    # sort photon_gm_datas according to y coordinate of center of mass
    photon_gm_datas.sort(key=lambda x: x.center_of_mass[1])
    photon_gm_datas.reverse()
    return photon_gm_datas


def calculate_photon_communities(
    gm_data: GMData, far_field_cutoff: float = 40, splitting_control: float = 0.4
) -> list[set[int]]:
    size = len(gm_data.means)
    adj_matrix_kl = np.zeros((size, size), dtype=np.float64)
    for i, (pos, cov) in enumerate(zip(gm_data.means, gm_data.covariances)):
        pos_ = pos / 100  # scale down position
        cov_ = cov / 10000  # scale down covariance
        for j, (pos_inner, cov_inner) in enumerate(
            zip(gm_data.means, gm_data.covariances)
        ):
            if i == j:
                continue
            pos_inner_ = pos_inner / 100  # scale down position
            cov_inner_ = cov_inner / 10000  # scale down covariance

            # symmetric KL divergence
            adj_matrix_kl[i, j] = kl_mvn(
                (pos_inner_, cov_inner_), (pos_, cov_)
            ) + kl_mvn(
                (pos_, cov_), (pos_inner_, cov_inner_)
            )  # +

    arr = far_field_cutoff - adj_matrix_kl
    arr[arr < 0] = 0
    arr = arr / np.max(arr)

    G = nx.from_numpy_array(arr)
    # print(G.edges(data=True))
    communities = nx.community.louvain_communities(G, resolution=splitting_control)
    return communities


# @dataclass
class PhotonGMData(BaseModel):
    center_of_mass: list[float]
    gm_photon: GMDataPhotonCommunity


def poisson(mean_photon_number, k, stop_at=5):
    if k < stop_at:
        return np.exp(-mean_photon_number) * mean_photon_number**k / math.factorial(k)
    if k >= stop_at:
        under = 0
        for j in range(stop_at):
            under += (
                np.exp(-mean_photon_number)
                * mean_photon_number**j
                / math.factorial(j)
            )

        return 1 - under


class StaticSimulation:
    """an object for storing and operating on a model of the detector response in 2D (t_a and t_b axese) as a function of mean photon number.
    It is created from a gaussian mixture model fit to the detector response at a particular mean photon number for which all the well-discernable
    and seperable photon number groupings can be identified and modeled.
    """

    def __init__(self, data: list[PhotonGMData]):
        self.data: list[PhotonGMData] = data
        self.single_photon_com = np.average(
            self.data[0].gm_photon.photon_means,
            weights=self.data[0].gm_photon.photon_weights,
            axis=0,
        )

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, "r") as f:
            data = orjson.loads(f.read())
            data = [PhotonGMData(**d) for d in data]
            return cls(data)

    def to_json(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(
                orjson.dumps(
                    [d.model_dump() for d in self.data],
                    option=orjson.OPT_SERIALIZE_NUMPY,
                ).decode()
            )

    def scale_simulation_to_match_mfr(self, mean_photon_rate) -> list[PhotonGMData]:
        sim_copy = copy.deepcopy(self.data)
        for i, community in enumerate(sim_copy):
            community.gm_photon.photon_weights = (
                community.gm_photon.photon_weights * poisson(mean_photon_rate, i + 1)
            )  # i+1 because photon number starts at 1

        return sim_copy

    def scale_mfr_and_apply_vector_offset(
        self, mean_photon_rate: float, offset: np.ndarray | list[float]
    ) -> list[PhotonGMData]:
        sim_copy = copy.deepcopy(self.data)
        for i, community in enumerate(sim_copy):
            community.gm_photon.photon_weights = (
                community.gm_photon.photon_weights * poisson(mean_photon_rate, i + 1)
            )
            community.gm_photon.photon_means = (
                community.gm_photon.photon_means - np.array(offset)
            )

        return sim_copy

    def flatten_sim_at_mfr(self, mean_photon_rate: float) -> GMData:
        res = self.scale_simulation_to_match_mfr(mean_photon_rate)

        weights = []
        means = []
        covariances = []
        for photon_community in res:
            weights.extend(photon_community.gm_photon.photon_weights)
            means.extend(photon_community.gm_photon.photon_means)
            covariances.extend(photon_community.gm_photon.photon_covariances)

        flattened_gm_data = GMData(
            weights=weights,
            means=means,
            covariances=covariances,
            num_components=len(weights),
            log_likelihood=0,
        )
        return flattened_gm_data


from matplotlib.patches import Ellipse

### colors, plotting, etc
import colorsys
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def mueller_colors(num_colors: int, start_hue: float = -0.4, gamma: float = 1.5, demo_visual = False, max_lightness: float = 1) -> list[str]:
    num_colors += 1
    hex_colors = []
    # Loop over num_colors values
    for i in range(num_colors):
        # Calculate lightness and hue
        lightness = (i / (num_colors - 1)) ** (1 / gamma)  # Apply gamma correction
        lightness *= max_lightness  # Scale to max_lightness
        hue = (i * 360 / (num_colors - 1) / 360 + start_hue) % 1  # Divide by 360 to normalize to [0, 1]
        # Convert HLS to RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, 1)
        # Convert RGB to hex and append to list
        hex_colors.append(colors.rgb2hex(rgb))

    if demo_visual:
        # Create a figure and a subplot
        fig, ax = plt.subplots(1, 1, figsize=(14, 2))
        # Create a bar plot with the colors
        for i, color in enumerate(hex_colors):
            ax.bar(i, 1, color=color)

        # Remove y-axis
        ax.get_yaxis().set_visible(False)

        # Set x-axis limits
        ax.set_xlim(-1, num_colors)

        # Set x-axis labels
        ax.set_xticks(range(num_colors))
        ax.set_xticklabels([f'Color {i+1}' for i in range(num_colors)])
        plt.show()
    return hex_colors


def draw_ellipse(position, covariance, ax=None, single_ellipse=False, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    try:
        ax = ax or plt.gca()
    except ValueError:
        print(
            "ax is not a valid matplotlib axis object. It might be an array of axese (not supported)"
        )

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    if single_ellipse:
        nsig = 3
        # print(kwargs)
        ax.add_patch(
            Ellipse(
                position, nsig * width, nsig * height, angle=angle, fill=False, **kwargs
            )
        )
    else:
        for nsig in np.linspace(0, 4, 8):
            ax.add_patch(
                Ellipse(
                    position,
                    nsig * width,
                    nsig * height,
                    angle=angle,
                    fill=False,
                    **kwargs
                )
            )


def plot_gmm(
    ax,
    g: GMData | GMDataPhotonCommunity,
    data_alpha=0.2,
    single_ellipse=False,
    weights_as_alpha=False,
    alpha_scaler: float = 5,
    **ellipse_kwargs
):
    """for plotting a all ellipses in a GMdata structure. single_ellipse controlls if one ellipse or more is plotted for each gaussian"""
    # w_factor = 0.2 / g.weights.max()

    if isinstance(g, GMData):
        max_weight = np.max(g.weights)
        label = ellipse_kwargs.pop('label', None)


        for pos, covar, w in zip(g.means, g.covariances, g.weights):
            if weights_as_alpha:
                if w == max_weight:
                    draw_ellipse(
                        pos,
                        covar,
                        ax=ax,
                        single_ellipse=single_ellipse,
                        alpha=min(1, w*alpha_scaler),
                        label=label,
                        **ellipse_kwargs
                    )
                else:
                    draw_ellipse(
                        pos,
                        covar,
                        ax=ax,
                        single_ellipse=single_ellipse,
                        alpha=min(1, w*alpha_scaler),
                        **ellipse_kwargs
                    )
            else:
                draw_ellipse(
                    pos, covar, ax=ax, single_ellipse=single_ellipse, label=label,**ellipse_kwargs
                )
                # only assign label to first ellipse
                label = None

    elif isinstance(g, GMDataPhotonCommunity) or isinstance(
        g, load_schema.GMDataPhotonCommunity
    ):
        max_weight = np.max(g.photon_weights)
        label = ellipse_kwargs.pop('label', None)


        for pos, covar, w in zip(
            g.photon_means, g.photon_covariances, g.photon_weights
        ):
            if weights_as_alpha:
                if w == max_weight:
                    draw_ellipse(
                        pos,
                        covar,
                        ax=ax,
                        single_ellipse=single_ellipse,
                        alpha=min(1, w*alpha_scaler),
                        label=label,
                        **ellipse_kwargs
                    )
                else:
                    draw_ellipse(
                        pos,
                        covar,
                        ax=ax,
                        single_ellipse=single_ellipse,
                        alpha=min(1, w*alpha_scaler),
                        **ellipse_kwargs
                    )
            else:
                draw_ellipse(
                    pos, covar, ax=ax, single_ellipse=single_ellipse, label=label, **ellipse_kwargs
                )
                # only assign label to first ellipse
                label = None
    else:
        print(type(g))
        raise TypeError("g must be of type GMData or GMDataPhotonCommunity")


def darken_and_saturate(hex_color, darken_factor=0.7, saturation_factor=1.3):
    # Convert hex to RGB
    rgb_color = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    rgb_color = [c / 255 for c in rgb_color]  # Normalize to [0, 1]

    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb_color)

    # Darken and saturate the color
    l = max(min(l * darken_factor, 1), 0)
    s = max(min(s * saturation_factor, 1), 0)

    # Convert back to RGB
    rgb_color = colorsys.hls_to_rgb(h, l, s)
    rgb_color = [int(c * 255) for c in rgb_color]  # Denormalize from [0, 1] to [0, 255]

    # Convert back to hex
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_color)

    return hex_color


def lighten_and_desaturate(hex_color, lighten_factor=1.3, desaturation_factor=0.7):
    # Convert hex to RGB
    rgb_color = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    rgb_color = [c / 255 for c in rgb_color]  # Normalize to [0, 1]

    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb_color)

    # Lighten and desaturate the color
    l = max(min(l * lighten_factor, 1), 0)
    s = max(min(s * desaturation_factor, 1), 0)

    # Convert back to RGB
    rgb_color = colorsys.hls_to_rgb(h, l, s)
    rgb_color = [int(c * 255) for c in rgb_color]  # Denormalize from [0, 1] to [0, 255]

    # Convert back to hex
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_color)

    return hex_color
