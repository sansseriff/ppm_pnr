import numpy as np

# from load_schema import *
from load_schema import Decode, Event, Result
from gmm_solver import (
    correction_from_gaussian_model,
    pnr_correction_from_gaussian_model,
)
from enum import Enum

from util import PhotonGMData
from pydantic import BaseModel


class DecodingChoice(Enum):
    GMM = 1
    SLOPE = 2


def decode_results(
    decode: Decode,
    choice: DecodingChoice,
    slice: int = 0,
    offset: float = 0.0,
    sim_data: list[PhotonGMData] | None = None,
) -> tuple[list[Event], list[Event]]:
    """loop over a list of events inside of Decode.results and check if the measurements
    based on either the SLOPE method or the GMM method are correct or incorrect.

    Args:
        decode (Decode): datastructure containing all data from the main PPM decoding process
        choice (DecodingChoice): decode using the GMM or the SLOPE method?
        slice (int, optional): Decode.results is a list of list. Each inner list is a full set of events
        to decode the image. There are multiple inner lists because each AWG sequence was repeated multiple times.
        offset (float, optional): Offset of the data w.r.t the clock.

    Returns:
        _type_: list of events updated with the new offset value. Same format style as the input Decode.results[0]
    """
    results = decode.results[slice]
    # if sim_data is None:
    gm_data = decode.gm_data.gm_list[
        10
    ]  # pick a reasonably high number of gaussians for the GMM
    new_results: list[Event] = []
    missing_results: list[Event] = []
    for event in results:
        if event.result != Result.MISSING:
            # then we have data to re-analyze with the gmm and a variable offset
            if choice == DecodingChoice.GMM:
                if sim_data is None:
                    # regular GMM decoding
                    gm_data = decode.gm_data.gm_list[
                        10
                    ]  # pick a reasonably high number of gaussians for the GMM
                    gauss_correction = correction_from_gaussian_model(
                        event.measured,
                        (event.tag_x, event.tag_y),
                        gm_data,
                        laser_time=50,
                        offset=offset,
                    )

                    gaussian_measured = event.measured + gauss_correction

                    if gaussian_measured == event.true:
                        res = Result.CORRECT
                    else:
                        res = Result.INCORRECT

                    new_results.append(
                        Event(
                            tag_x=event.tag_x,
                            tag_y=event.tag_y,
                            tag=event.tag,
                            true=event.true,
                            measured=event.measured,
                            gaussian_measured=gaussian_measured,
                            result=res,
                        )
                    )
                else:
                    # pnr GMM decoding
                    gm_data = sim_data
                    ###
                    pnr_best, pnr_2nd_best = pnr_correction_from_gaussian_model(
                        event.measured,
                        (event.tag_x, event.tag_y),
                        gm_data,
                        laser_time=50,
                        offset=offset,
                    )

                    gaussian_measured = event.measured + pnr_best.correction

                    if gaussian_measured == event.true:
                        res = Result.CORRECT
                    else:
                        res = Result.INCORRECT

                    # update pnr_best and pnr_2nd_best with new measured value
                    pnr_best.measured = event.measured + pnr_best.correction
                    pnr_2nd_best.measured = event.measured + pnr_2nd_best.correction

                    new_results.append(
                        Event(
                            tag_x=event.tag_x,
                            tag_y=event.tag_y,
                            tag=event.tag,
                            true=event.true,
                            measured=event.measured,
                            gaussian_measured=gaussian_measured,
                            result=res,
                            pnr_best=pnr_best,
                            pnr_2nd_best=pnr_2nd_best,
                        )
                    )
            elif choice == DecodingChoice.SLOPE:
                new_slope_solution = round((event.tag + offset) / 50)

                if new_slope_solution == event.true:
                    res = Result.CORRECT
                else:
                    res = Result.INCORRECT

                new_results.append(
                    Event(
                        tag_x=event.tag_x,
                        tag_y=event.tag_y,
                        tag=event.tag,
                        true=event.true,
                        measured=event.measured,
                        gaussian_measured=event.gaussian_measured,
                        result=res,
                    )
                )
        else:
            missing_results.append(event)

    return new_results, missing_results


class MultiprocessLoaderCorrector:
    def __init__(self):
        pass

    def __call__(self, decode: Decode):
        return caller(decode)


class Res(BaseModel):
    empty: list[Event]
    filled: list[Event]


class CallerResults(BaseModel):
    max_offset: float | None = None
    list_cc: list | None = None
    list_org_cc: list | None = None
    offsets: list | None = None
    gmm_optimized_results: Res | None = None
    slope_optimized_results: Res | None = None


class DecodingResults(BaseModel):
    list_caller_results: list[CallerResults]


def stack_decode_results(
    decode: Decode,
    choice: DecodingChoice,
    stack: int,
    offset: float = 0.0,
    sim_data: list[PhotonGMData] = None,
):
    """decode multiple stacks with custom offset value. This is used to find the optimal offset value in
    the caller function. Events from the added slices are just appended to the end of the output list.
    Because this list is just used to check relative decoding success fraction.

    Args:
        decode (Decode): datastructure containing all data from the main PPM decoding process
        choice (DecodingChoice): decode using the GMM or the SLOPE method?
        stack (int): number of slices to decode
        offset (float, optional): Offset of the data w.r.t the clock. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    new_results = []
    missing_results = []
    for sl in range(stack):
        new, missing = decode_results(
            decode, choice=choice, slice=sl, offset=offset, sim_data=sim_data
        )
        new_results.extend(new)
        missing_results.extend(missing)
    return new_results, missing_results


def caller(decode: Decode):
    list_cc = []
    list_org_cc = []
    max_offsets_vs_dB = []
    stack = 3

    offsets = np.arange(-0.19, 0.04, 0.0025).tolist()
    for offset in offsets:
        # new_results, missing_results = decode_results(decode, choice = DecodingChoice.GMM, slice=0, offset=offset)
        new_results, missing_results = stack_decode_results(
            decode, choice=DecodingChoice.GMM, stack=stack, offset=offset
        )
        cc = 0
        for event in new_results:
            if event.result == Result.CORRECT:
                cc += 1

        # new_results, missing_results = decode_results(decode, choice = DecodingChoice.SLOPE, slice=0, offset=0)
        new_results, missing_results = stack_decode_results(
            decode, choice=DecodingChoice.SLOPE, stack=stack, offset=0
        )
        org_cc = 0
        for event in new_results:
            if event.result == Result.CORRECT:
                org_cc += 1

        cc = cc / len(new_results)
        org_cc = org_cc / len(new_results)
        list_cc.append(cc)
        list_org_cc.append(org_cc)
        print(f"cc: {round(cc,4)}, org_cc: {round(org_cc,4)}, offset: {offset}")

    max_offset = offsets[np.argmax(list_cc)]
    max_offset_regular = offsets[np.argmax(list_org_cc)]
    max_offsets_vs_dB.append(max_offset)

    new_results, missing_results = stack_decode_results(
        decode, choice=DecodingChoice.GMM, stack=stack, offset=max_offset
    )

    res_gmm = Res(empty=missing_results, filled=new_results)

    new_results, missing_results = stack_decode_results(
        decode, choice=DecodingChoice.SLOPE, stack=stack, offset=max_offset_regular
    )

    res_slope = Res(empty=missing_results, filled=new_results)

    return CallerResults(
        max_offset=max_offset,
        list_cc=list_cc,
        list_org_cc=list_org_cc,
        offsets=offsets,
        gmm_optimized_results=res_gmm,
        slope_optimized_results=res_slope,
    )


def pnr_caller(data: tuple[Decode, list[PhotonGMData], float | None]):
    decode: Decode = data[0]
    sim_data: list[PhotonGMData] = data[1]
    offset_param = data[2] if len(data) > 2 else None

    print("this is offset param: ", offset_param)

    if offset_param is not None:
        stack = 5
        new_results, missing_results = stack_decode_results(
            decode,
            choice=DecodingChoice.GMM,
            stack=stack,
            offset=offset_param,
            sim_data=sim_data,
        )
        res_gmm = Res(empty=missing_results, filled=new_results)
        return CallerResults(max_offset=offset_param, gmm_optimized_results=res_gmm)

    list_cc = []
    list_org_cc = []
    max_offsets_vs_dB = []
    stack = 3

    offsets = np.arange(-0.19, 0.04, 0.0025).tolist()
    for offset in offsets:
        # new_results, missing_results = decode_results(decode, choice = DecodingChoice.GMM, slice=0, offset=offset)
        new_results, missing_results = stack_decode_results(
            decode,
            choice=DecodingChoice.GMM,
            stack=stack,
            offset=offset,
            sim_data=sim_data,
        )
        cc = 0
        for event in new_results:
            if event.result == Result.CORRECT:
                cc += 1

        # new_results, missing_results = decode_results(decode, choice = DecodingChoice.SLOPE, slice=0, offset=0)
        new_results, missing_results = stack_decode_results(
            decode,
            choice=DecodingChoice.SLOPE,
            stack=stack,
            offset=0,
            sim_data=sim_data,
        )
        org_cc = 0
        for event in new_results:
            if event.result == Result.CORRECT:
                org_cc += 1

        cc = cc / len(new_results)
        org_cc = org_cc / len(new_results)
        list_cc.append(cc)
        list_org_cc.append(org_cc)
        print(f"cc: {round(cc,4)}, org_cc: {round(org_cc,4)}, offset: {offset}")

    max_offset = offsets[np.argmax(list_cc)]
    max_offset_regular = offsets[np.argmax(list_org_cc)]
    max_offsets_vs_dB.append(max_offset)

    new_results, missing_results = stack_decode_results(
        decode, choice=DecodingChoice.GMM, stack=stack, offset=max_offset
    )

    res_gmm = Res(empty=missing_results, filled=new_results)

    new_results, missing_results = stack_decode_results(
        decode, choice=DecodingChoice.SLOPE, stack=stack, offset=max_offset_regular
    )

    res_slope = Res(empty=missing_results, filled=new_results)

    return CallerResults(
        max_offset=max_offset,
        list_cc=list_cc,
        list_org_cc=list_org_cc,
        offsets=offsets,
        gmm_optimized_results=res_gmm,
        slope_optimized_results=res_slope,
    )


class NumpyConvertingStruct:
    def __init__(self):
        pass

    def numpyify(self):
        for item in self.__dict__:
            self.__dict__[item] = np.array(self.__dict__[item])


class OutExport(NumpyConvertingStruct):
    def __init__(self):
        self.correct_gmm: list | np.ndarray = []
        self.incorrect_gmm: list | np.ndarray = []
        self.correct_slope: list | np.ndarray = []
        self.incorrect_slope: list | np.ndarray = []
        self.missing: list | np.ndarray = []
        self.lengths: list | np.ndarray = []
        self.lengths_detected: list | np.ndarray = []


def export_plot_metrics(dec: DecodingResults):
    s = OutExport()

    for dB_data in dec.list_caller_results:
        total_results_gmm_optimized = (
            dB_data.gmm_optimized_results.filled + dB_data.gmm_optimized_results.empty
        )

        total_results_slope_optimized = (
            dB_data.slope_optimized_results.filled
            + dB_data.slope_optimized_results.empty
        )

        c_gmm = 0
        inc_gmm = 0

        c_slope = 0
        inc_slope = 0

        mis = 0

        for res in total_results_gmm_optimized:
            if res.result == Result.CORRECT:
                c_gmm += 1
            if res.result == Result.INCORRECT:
                inc_gmm += 1
            if res.result == Result.MISSING:
                mis += 1

        for res in total_results_slope_optimized:
            if res.result == Result.CORRECT:
                c_slope += 1
            if res.result == Result.INCORRECT:
                inc_slope += 1

        s.correct_gmm.append(c_gmm)
        s.incorrect_gmm.append(inc_gmm)

        s.correct_slope.append(c_slope)
        s.incorrect_slope.append(inc_slope)
        s.missing.append(mis)

        s.lengths.append(len(total_results_gmm_optimized))
        s.lengths_detected.append(len(dB_data.gmm_optimized_results.filled))

    s.numpyify()
    s.correct_gmm_err = np.sqrt(s.correct_gmm) / s.lengths_detected
    s.correct_gmm = s.correct_gmm / s.lengths

    s.incorrect_gmm_err = np.sqrt(s.incorrect_gmm) / s.lengths_detected
    s.incorrect_gmm = s.incorrect_gmm / s.lengths

    s.correct_slope = s.correct_slope / s.lengths
    s.incorrect_slope = s.incorrect_slope / s.lengths
    s.missing = s.missing / s.lengths

    return s


class OutDetectedExport(NumpyConvertingStruct):
    def __init__(self):
        self.correct_gmm_d: list | np.ndarray = []
        self.incorrect_gmm_d: list | np.ndarray = []
        self.correct_slope_d: list | np.ndarray = []
        self.incorrect_slope_d: list | np.ndarray = []
        self.lengths_detected: list | np.ndarray = []
        self.correct_gmm_d_err: list | np.ndarray = []
        self.incorrect_gmm_d_err: list | np.ndarray = []
        self.correct_slope_d_err: list | np.ndarray = []
        self.incorrect_slope_d_err: list | np.ndarray = []
        self.decoding_error_rate_improvement: list | np.ndarray = []
        self.decoding_error_rate_improvement_err: list | np.ndarray = []


def export_detected_plot_metrics(dec: DecodingResults):
    s = OutDetectedExport()

    for dB_data in dec.list_caller_results:
        c_gmm = 0
        inc_gmm = 0

        c_slope = 0
        inc_slope = 0

        mis = 0
        for res in dB_data.gmm_optimized_results.filled:
            if res.result == Result.CORRECT:
                c_gmm += 1
            if res.result == Result.INCORRECT:
                inc_gmm += 1
            if res.result == Result.MISSING:
                mis += 1

        for res in dB_data.slope_optimized_results.filled:
            if res.result == Result.CORRECT:
                c_slope += 1
            if res.result == Result.INCORRECT:
                inc_slope += 1

        s.correct_gmm_d.append(c_gmm)
        s.incorrect_gmm_d.append(inc_gmm)

        s.correct_slope_d.append(c_slope)
        s.incorrect_slope_d.append(inc_slope)
        s.lengths_detected.append(len(dB_data.gmm_optimized_results.filled))

    s.numpyify()

    s.correct_gmm_d_err = np.sqrt(
        (np.sqrt(s.correct_gmm_d) / s.lengths_detected) ** 2
        + (np.sqrt(s.lengths_detected) * s.correct_gmm_d / (s.lengths_detected**2)) ** 2
    )
    s.correct_gmm_d = s.correct_gmm_d / s.lengths_detected

    s.incorrect_gmm_d_err = np.sqrt(
        (np.sqrt(s.incorrect_gmm_d) / s.lengths_detected) ** 2
        + (np.sqrt(s.lengths_detected) * s.incorrect_gmm_d / (s.lengths_detected**2))
        ** 2
    )
    s.incorrect_gmm_d = s.incorrect_gmm_d / s.lengths_detected

    # the error of a/b
    s.correct_slope_d_err = np.sqrt(
        (np.sqrt(s.correct_slope_d) / s.lengths_detected) ** 2
        + (np.sqrt(s.lengths_detected) * s.correct_slope_d / (s.lengths_detected**2))
        ** 2
    )
    # evaluate a/b
    s.correct_slope_d = s.correct_slope_d / s.lengths_detected

    s.incorrect_slope_d_err = np.sqrt(
        (np.sqrt(s.incorrect_slope_d) / s.lengths_detected) ** 2
        + (np.sqrt(s.lengths_detected) * s.incorrect_slope_d / (s.lengths_detected**2))
        ** 2
    )
    s.incorrect_slope_d = s.incorrect_slope_d / s.lengths_detected

    a = s.incorrect_slope_d
    b = s.incorrect_gmm_d
    da = s.incorrect_slope_d_err
    db = s.incorrect_gmm_d_err
    s.decoding_error_rate_improvement = (a - b) / a
    s.decoding_error_rate_improvement_err = np.sqrt(
        (b * da / (a**2)) ** 2 + (-db / a) ** 2
    )

    return s


class OutExport2D(NumpyConvertingStruct):
    def __init__(self):
        self.gmm_correct_slope_incorrect: list | np.ndarray = []
        self.gmm_incorrect_slope_correct: list | np.ndarray = []
        self.both_correct: list | np.ndarray = []
        self.gmm_correct: list | np.ndarray = []
        self.slope_correct: list | np.ndarray = []


def export_2d_data(caller_res: CallerResults):
    s = OutExport2D()

    caller_res.gmm_optimized_results

    for event_gmm, event_slope in zip(
        caller_res.gmm_optimized_results.filled,
        caller_res.slope_optimized_results.filled,
    ):
        if (
            event_gmm.result == Result.CORRECT
            and event_slope.result == Result.INCORRECT
        ):
            s.gmm_correct_slope_incorrect.append(
                (
                    event_gmm.tag_x - 50 * event_gmm.true,
                    event_gmm.tag_y - 50 * event_gmm.true,
                )
            )

        if (
            event_gmm.result == Result.INCORRECT
            and event_slope.result == Result.CORRECT
        ):
            s.gmm_incorrect_slope_correct.append(
                (
                    event_gmm.tag_x - 50 * event_gmm.true,
                    event_gmm.tag_y - 50 * event_gmm.true,
                )
            )

        if event_gmm.result == Result.CORRECT and event_slope.result == Result.CORRECT:
            s.both_correct.append(
                (
                    event_gmm.tag_x - 50 * event_gmm.true,
                    event_gmm.tag_y - 50 * event_gmm.true,
                )
            )

        if event_gmm.result == Result.CORRECT:
            s.gmm_correct.append(
                (
                    event_gmm.tag_x - 50 * event_gmm.true,
                    event_gmm.tag_y - 50 * event_gmm.true,
                )
            )

        if event_slope.result == Result.CORRECT:
            s.slope_correct.append(
                (
                    event_slope.tag_x - 50 * event_slope.true,
                    event_slope.tag_y - 50 * event_slope.true,
                )
            )

    s.numpyify()
    return s
