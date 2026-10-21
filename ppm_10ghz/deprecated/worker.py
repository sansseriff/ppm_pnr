import numpy as np
from load_schema import *
from gmm_solver import correction_from_gaussian_model
from enum import Enum

class DecodingChoice(Enum):
    GMM = 1
    SLOPE = 2





def decode_results(decode: Decode, choice: DecodingChoice, laser_time: float, slice=0, offset: float = 0.0):
    results = decode.results[slice]
    gm_data = decode.gm_data.gm_list[9]

    new_results = []
    missing_results = []
    for event in results:
        
        if event.result != Result.MISSING:
            # then we have data to re-analyze with the gmm and a variable offset
            if choice == DecodingChoice.GMM:
                gauss_correction = correction_from_gaussian_model(
                    event.measured,
                    (event.tag_x, event.tag_y),
                    gm_data,
                    laser_time=laser_time,
                    offset=offset,
                )

                gaussian_measured = event.measured + gauss_correction

                if gaussian_measured == event.true:
                    res = Result.CORRECT
                if gaussian_measured != event.true:
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

            elif choice == DecodingChoice.SLOPE:
                new_slope_solution = round((event.tag + offset) / laser_time)

                if new_slope_solution == event.true:
                    res = Result.CORRECT
                if new_slope_solution != event.true:
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
    max_offset: float
    list_cc: list
    list_org_cc: list
    offsets: list
    gmm_optimized_results: Res
    slope_optimized_results: Res


class DecodingResults(BaseModel):
        list_caller_results: list[CallerResults]


def stack_decode_results(decode: Decode, choice: DecodingChoice, laser_time: float, stack: tuple, offset: float = 0.0):
    new_results = []
    missing_results = []
    for sl in range(*stack):
        new, missing = decode_results(decode, choice, laser_time, slice=sl, offset=offset)
        new_results.extend(new)
        missing_results.extend(missing)
    return new_results, missing_results


def caller(decode: Decode):
    list_cc = []
    list_org_cc = []
    max_offsets_vs_dB = []
    stack = (0,2)

    laser_time = decode.set_data["laser_time"]*1e12

    offsets = np.arange(-0.19, 0.04, 0.0025).tolist()
    for offset in offsets:
        # new_results, missing_results = decode_results(decode, choice = DecodingChoice.GMM, slice=0, offset=offset)
        new_results, missing_results = stack_decode_results(decode, DecodingChoice.GMM, laser_time, stack=stack, offset=offset)
        cc = 0
        for event in new_results:
            if event.result == Result.CORRECT:
                cc += 1

        # new_results, missing_results = decode_results(decode, choice = DecodingChoice.SLOPE, slice=0, offset=0)
        new_results, missing_results = stack_decode_results(decode, DecodingChoice.SLOPE, laser_time, stack=stack, offset=0)
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

    new_results, missing_results = stack_decode_results(decode, DecodingChoice.GMM, laser_time, stack=stack, offset=max_offset)

    res_gmm = Res(empty=missing_results, filled=new_results)

    new_results, missing_results = stack_decode_results(decode, DecodingChoice.SLOPE, laser_time, stack=stack, offset=max_offset_regular)

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
        total_results_gmm_optimized = dB_data.gmm_optimized_results.filled + dB_data.gmm_optimized_results.empty

        total_results_slope_optimized = dB_data.slope_optimized_results.filled + dB_data.slope_optimized_results.empty

        c_gmm = 0
        inc_gmm = 0

        c_slope = 0
        inc_slope = 0

        mis = 0

        for res in total_results_gmm_optimized:
            if res.result == Result.CORRECT: c_gmm += 1
            if res.result == Result.INCORRECT: inc_gmm += 1
            if res.result == Result.MISSING: mis += 1

        for res in total_results_slope_optimized:
            if res.result == Result.CORRECT: c_slope += 1
            if res.result == Result.INCORRECT: inc_slope += 1
            

        s.correct_gmm.append(c_gmm)
        s.incorrect_gmm.append(inc_gmm)

        s.correct_slope.append(c_slope)
        s.incorrect_slope.append(inc_slope)
        s.missing.append(mis)

        s.lengths.append(len(total_results_gmm_optimized))
        s.lengths_detected.append(len(dB_data.gmm_optimized_results.filled))


    s.numpyify()
    s.correct_gmm_err = np.sqrt(s.correct_gmm)/s.lengths_detected
    s.correct_gmm = s.correct_gmm/s.lengths

    s.incorrect_gmm_err = np.sqrt(s.incorrect_gmm)/s.lengths_detected
    s.incorrect_gmm = s.incorrect_gmm/s.lengths

    s.correct_slope = s.correct_slope/s.lengths
    s.incorrect_slope = s.incorrect_slope/s.lengths
    s.missing = s.missing/s.lengths

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
            if res.result == Result.CORRECT: c_gmm += 1
            if res.result == Result.INCORRECT: inc_gmm += 1
            if res.result == Result.MISSING: mis += 1

        for res in dB_data.slope_optimized_results.filled:
            if res.result == Result.CORRECT: c_slope += 1
            if res.result == Result.INCORRECT: inc_slope += 1

        s.correct_gmm_d.append(c_gmm)
        s.incorrect_gmm_d.append(inc_gmm)
        
        s.correct_slope_d.append(c_slope)
        s.incorrect_slope_d.append(inc_slope)
        s.lengths_detected.append(len(dB_data.gmm_optimized_results.filled))

    s.numpyify()

    s.correct_gmm_d_err = np.sqrt((np.sqrt(s.correct_gmm_d)/s.lengths_detected)**2 + (np.sqrt(s.lengths_detected)*s.correct_gmm_d/(s.lengths_detected**2))**2)
    s.correct_gmm_d = s.correct_gmm_d/s.lengths_detected

    s.incorrect_gmm_d_err = np.sqrt((np.sqrt(s.incorrect_gmm_d)/s.lengths_detected)**2 + (np.sqrt(s.lengths_detected)*s.incorrect_gmm_d/(s.lengths_detected**2))**2)
    s.incorrect_gmm_d = s.incorrect_gmm_d/s.lengths_detected

    # the error of a/b
    s.correct_slope_d_err = np.sqrt((np.sqrt(s.correct_slope_d)/s.lengths_detected)**2 + (np.sqrt(s.lengths_detected)*s.correct_slope_d/(s.lengths_detected**2))**2)
    # evaluate a/b
    s.correct_slope_d = s.correct_slope_d/s.lengths_detected



    s.incorrect_slope_d_err = np.sqrt((np.sqrt(s.incorrect_slope_d)/s.lengths_detected)**2 + (np.sqrt(s.lengths_detected)*s.incorrect_slope_d/(s.lengths_detected**2))**2)
    s.incorrect_slope_d = s.incorrect_slope_d/s.lengths_detected

    a = s.incorrect_slope_d
    b = s.incorrect_gmm_d
    da = s.incorrect_slope_d_err
    db = s.incorrect_gmm_d_err
    s.decoding_error_rate_improvement = (a - b)/a
    s.decoding_error_rate_improvement_err = np.sqrt((b*da/(a**2))**2 + (-db/a)**2)

    return s


class OutExport2D(NumpyConvertingStruct):
        def __init__(self):
            self.gmm_correct_slope_incorrect: list | np.ndarray = []
            self.gmm_incorrect_slope_correct: list | np.ndarray = []
            self.both_correct: list | np.ndarray = []
            self.gmm_correct: list | np.ndarray = []
            self.slope_correct: list | np.ndarray = []

def export_2d_data(caller_res: CallerResults, laser_time: float):

    s = OutExport2D()

    caller_res.gmm_optimized_results
    
    for event_gmm, event_slope in zip(caller_res.gmm_optimized_results.filled, caller_res.slope_optimized_results.filled):
        if event_gmm.result == Result.CORRECT and event_slope.result == Result.INCORRECT:
            s.gmm_correct_slope_incorrect.append((event_gmm.tag_x - laser_time*event_gmm.true, event_gmm.tag_y - laser_time*event_gmm.true))

        if event_gmm.result == Result.INCORRECT and event_slope.result == Result.CORRECT:
            s.gmm_incorrect_slope_correct.append((event_gmm.tag_x - laser_time*event_gmm.true, event_gmm.tag_y - laser_time*event_gmm.true))

        if event_gmm.result == Result.CORRECT and event_slope.result == Result.CORRECT:
            s.both_correct.append((event_gmm.tag_x - laser_time*event_gmm.true, event_gmm.tag_y - laser_time*event_gmm.true))

        if event_gmm.result == Result.CORRECT:
            s.gmm_correct.append((event_gmm.tag_x - laser_time*event_gmm.true, event_gmm.tag_y - laser_time*event_gmm.true))

        if event_slope.result == Result.CORRECT:
            s.slope_correct.append((event_slope.tag_x - laser_time*event_slope.true, event_slope.tag_y - laser_time*event_slope.true))


    s.numpyify()
    return s
        