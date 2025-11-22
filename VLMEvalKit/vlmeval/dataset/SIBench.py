from .image_base import ImageBaseDataset
from .image_mcq import ImageMCQDataset
from .video_base import VideoBaseDataset
from ..smp import *
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import can_infer, track_progress_rich
from .utils.yorn import YOrN_match_prompt, YOrN_Extraction
import os
import decord
import re


def build_prompt(question, prediction):
    tmpl = (
        "You are an AI assistant who will help me to match "
        "an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, "
        "and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
        "Example 1: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: a cute teddy bear\nYour output: A\n"
        "Example 2: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: Spider\nYour output: Z\n"
        "Example 3: \n"
        "Question and Options: {}?\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, prediction)


FAIL_MSG = "Failed to obtain answer via API."


def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_Number_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line["question"]
    prediction = str(line["prediction"])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + "\n"
    prompt += "Question: " + question + "\n"
    prompt += "Model respone: " + prediction
    prompt += "Extracted answer:"
    return prompt


def extract_answer_from_item(model, item, dataset_name=None):
    logger = get_logger("Evaluation")
    # It will return: (pred, raw, llm_time)
    if item["type"] == "MCQ":
        prompt = build_prompt(item["question"], item["prediction"])

        if model is None:
            return dict(
                res="Z",
                log="Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.",
            )

        retry = 3
        while retry:
            ans = model.generate(prompt).strip()
            chars = ".()[],:;!*#{}"
            for c in chars:
                ans = ans.replace(c, " ")
            if "Failed to obtain answer via API" in ans:
                logger.warning("GPT API failed to answer. ")
            else:
                if len(ans) == 1:
                    return dict(res=ans, log=ans)
                else:
                    logger.warning(
                        f"Output includes 0 / > 1 letter among candidates and Z: {ans}"
                    )
            retry -= 1

            if retry == 0:
                return dict(
                    res="Z",
                    log="Failed to predict.",
                )
    elif item["type"] == "YN":
        prompt = YOrN_match_prompt(item)
        retry = 5
        for i in range(retry):
            output = model.generate(prompt, temperature=0.5 * i)
            ans = YOrN_Extraction(output)
            if ans != "Unknown":
                return dict(res=ans, log=ans)
        return dict(
            res="Unknown",
            log="Failed to predict.",
        )
    elif item["type"] == "Number" or item["type"] == "Number_Int":
        prompt = build_Number_prompt(item)
        log = ""
        retry = 5
        for i in range(retry):
            prediction = item["prediction"]
            res = model.generate(prompt, temperature=i * 0.5)

            if FAIL_MSG in res:
                log += f"Try {i}: output is {prediction}, failed to parse.\n"
            else:
                log += "Succeed"
                return dict(log=log, res=res.strip().lower())
        log += "All 5 retries failed.\n"
        return dict(log=log, res="")
    else:
        raise ValueError(f"Unknown type {item['type']}")


def compute_mra(y_true, y_pred):
    C = np.arange(0.5, 1.0, 0.05)
    mra_sum = 0
    for theta in C:
        relative_error = np.abs(y_pred - y_true) / y_true
        if relative_error < (1 - theta):
            mra_sum += 1
    mra = mra_sum / len(C)
    return mra


def eval_vanilla(model, item, dataset_name=None):
    res = extract_answer_from_item(model, item, dataset_name=dataset_name)
    res, match_log = res["res"], res["log"]
    if item["type"] == "Number":
        try:
            pred = eval(res.strip().lower())
            gt = eval(item["GT"].strip().lower())
            mra = compute_mra(gt, pred)
            return dict(hit=mra, log=f"Match Log: {match_log}. ")
        except Exception as e:
            print(e)
            return dict(hit=0, log=f"Match Log: {match_log}. ")
    else:
        if res.strip().lower() == item["GT"].strip().lower():
            return dict(hit=1, log=f"Match Log: {match_log}. ")
        else:
            return dict(hit=0, log=f"Match Log: {match_log}. ")


# data, meta are pd.DataFrame, result_file is a path
def sibench_vanilla_eval(model, data, meta, nproc, result_file, dataset_name=None):
    result = {}
    if osp.exists(result_file):
        result = load(result_file)
    answer_map = {i: c for i, c in zip(meta["index"], meta["answer"])}

    data = data[data["index"].isin(answer_map)]
    data["GT"] = [answer_map[idx] for idx in data["index"]]
    items = []

    for i in range(len(data)):
        # Dealing with the normal part
        item = data.iloc[i]
        if item["index"] not in result:
            items.append(item)

    tups = [dict(model=model, item=x, dataset_name=dataset_name) for x in items]
    keys = [x["index"] for x in items]
    if len(tups):
        res = track_progress_rich(
            eval_vanilla,
            tups,
            nproc=nproc,
            chunksize=nproc,
            save=result_file,
            keys=keys,
        )
        result = load(result_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v
    data["hit"] = [result[i]["hit"] for i in data["index"]]
    data["log"] = [result[i]["log"] for i in data["index"]]
    if "GT" in data:
        data.pop("GT")
    return data


class SIBench(ImageMCQDataset, ImageBaseDataset, VideoBaseDataset):
    MODALITY = "MixedInput"
    TYPE = "MixedOutput"

    NEED_EXTRA_PROMPT_SOURCE = [
        "vstibench",
        "MMSI-Bench",
        "3DSRBench",
        "OmniSpatial",
        "Spatial-MM",
        "SpatialMQA",
        "VSI-Bench",
        "STI-Bench",
        "SpatialEval",
        "SITE-Bench",
        "SPHERE-VLM",
        "SRBench",
        "BLINK",
    ]
    # do not need = SpatialBench, SPAR-Bench, Super-CLEVR-3D, Omni3D-Bench
    SETTING = [
        "relative_distance",
        "Reach_Prediction",
        "Object_Shape",
        "Height",
        "Existence",
        "Spatial_Compatibility",
        "Coordinate_Conversion",
        "Counting",
        "Route_Planning",
        "Trajectory_Description",
        "Geometric_Reasoning",
        "Spatial_Imagination",
        "Object_Size_Estimation",
        "Spatial_Grid",
        "Situational_QA",
        "Velocity_Acceleration",
        "Maze_Navigation",
        "Temporal-Appearance_Order",
        "Camera_Pose",
        "Occlusion",
        "multi-view_reasoning",
        "Object_Localization",
        "Spatial_Relation",
    ]

    # Counting Camera_Pose Coordinate_Conversion multi-view_reasoning Object_Shape Object_Size_Estimation Occlusion relative_distance Situational_QA Spatial_Grid Spatial_Relation Trajectory_Description
    # Reach_Prediction Height Existence Spatial_Compatibility Route_Planning Geometric_Reasoning Velocity_Acceleration Spatial_Imagination Temporal-Appearance_Order Object_Localization
    VIDEO_MODALITY_INCLUDED_SETTING = [""]

    FRAMES_TMPL_SYS = """
You will receive {} distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
Please analyze these frames and answer the question based on your observations.
"""
    FRAMES_TMPL_SYS_4VIDEO_LLM = """
You will receive several distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
Please analyze these frames and answer the question based on your observations.
"""

    def __init__(self, dataset="MMBench", skip_noimg=True, nframe=30, fps=-1):
        super(SIBench, self).__init__(dataset, skip_noimg)

        self.frame_tmpl = "frame-{}-of-{}.jpg"
        self.frame_tmpl_fps = "frame-{}-of-{}-{}fps.jpg"

        self.nframe = nframe
        self.fps = fps
        if self.fps > 0 and self.nframe > 0:
            raise ValueError("fps and nframe should not be set at the same time")
        if self.fps <= 0 and self.nframe <= 0:
            raise ValueError("fps and nframe should be set at least one valid value")

    @classmethod
    def supported_datasets(cls):
        return cls.SETTING

    def add_extra_prompt(self, prompt, answer_type, data_source):
        if data_source in self.NEED_EXTRA_PROMPT_SOURCE:
            if answer_type == "MCQ":
                prompt += "\nSelect from the given options, answer with letters only."
            elif answer_type == "YN":
                prompt += "\nAnswer with 'Yes' or 'No' only."
            elif answer_type.startswith("Number"):
                prompt += "\nAnswer using a single number and nothing else."
            else:
                raise NotImplementedError(
                    f"Answer type '{answer_type}' is not supported. Supported types are: 'MCQ', 'YN', 'Number'."
                )
        elif data_source is None:
            raise KeyError("Required key 'data_source' is missing.")
        return prompt

    def frame_paths(self, video, data_base):
        # need self.frame_root & self.frame_tmpl & self.nframe
        frame_root = osp.join(data_base, video.split("/")[0], "frames")
        os.makedirs(frame_root, exist_ok=True)
        return [
            osp.join(frame_root, self.frame_tmpl.format(i, self.nframe))
            for i in range(1, self.nframe + 1)
        ]

    def save_video_frames(self, line, data_base):
        # need self.nframe & self.fps
        video = line["video_path"]
        vid_path = os.path.normpath(os.path.join(data_base, line["video_path"]))
        vid = decord.VideoReader(vid_path)
        video_info = {
            "fps": vid.get_avg_fps(),
            "n_frames": len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video, data_base)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info["n_frames"] / video_info["fps"]
            required_frames = int(total_duration * self.fps)
            step_size = video_info["fps"] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    def save_video_into_images(self, line, data_base):
        frame_paths = self.save_video_frames(line, data_base)
        return frame_paths

    def build_prompt_for_video(self, line, video_llm, data_base):
        # need video_llm
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_path = os.path.normpath(os.path.join(data_base, line["video_path"]))
        prompt = line["question"]
        answer_type = line.get("type")
        data_source = line.get("data_source")
        prompt = self.add_extra_prompt(prompt, answer_type, data_source)

        if video_llm:  # video_llm
            message = [dict(type="text", value=self.FRAMES_TMPL_SYS_4VIDEO_LLM)]
            message.append(dict(type="text", value=prompt))
            message.append(dict(type="video", value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line, data_base)
            message = [
                dict(
                    type="text", value=self.FRAMES_TMPL_SYS.format(len(img_frame_paths))
                )
            ]
            message.append(dict(type="text", value=prompt))
            for im in img_frame_paths:
                message.append(dict(type="image", value=im))
        return message

    def build_prompt_for_image(self, line, data_base):
        msgs = []
        if line.get("image_path"):
            tgt_path = toliststr("".join(line["image_path"].split()).split(","))
            for _ in range(len(tgt_path)):
                tgt_path[_] = os.path.join(data_base, tgt_path[_])
        else:
            raise KeyError("Required key 'image_path' is missing.")

        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]

        question = line["question"]
        prompt = question
        answer_type = line.get("type")
        data_source = line.get("data_source")
        prompt = self.add_extra_prompt(prompt, answer_type, data_source)
        msgs.append(dict(type="text", value=prompt))
        return msgs

    def build_prompt(self, line, video_llm=None, data_base="."):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if line.get("input_type") in ["image", "multi-view"]:
            return self.build_prompt_for_image(line=line, data_base=data_base)
        elif line.get("input_type") == "video":
            video_data_base = data_base.replace("/data", "/data_sampled_video")
            return self.build_prompt_for_video(
                line=line, video_llm=video_llm, data_base=video_data_base
            )
        else:
            raise NotImplementedError(
                f"Unrecognized input type: {line.get('input_type')}.\
                                       Just support 'image', 'multi-view' and 'video'."
            )

    def extract_numbers_from_string(self, text, reverse_order):
        number_strings = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
        result = []
        for num_str in number_strings:
            cleaned_str = num_str.replace(",", "")
            try:
                result.append(float(cleaned_str))
            except ValueError:
                continue

        if reverse_order:
            result.reverse()

        return result

    def compute_mra(self, y_true, y_pred):
        C = np.arange(0.5, 1.0, 0.05)
        mra_sum = 0
        for theta in C:
            relative_error = np.abs(y_pred - y_true) / y_true
            if relative_error < (1 - theta):
                mra_sum += 1
        mra = mra_sum / len(C)
        return mra

    def yn_Extraction(self, pred):
        pred = pred.strip().lower()
        pred = re.sub(r"[^\w\s]", "", pred)

        if pred == "yes":
            return "yes"
        elif pred == "no":
            return "no"
        else:
            return pred

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import extract_characters_regex, report_acc
        from .utils.yorn import YOrN_Extraction

        assert eval_file.endswith(".xlsx"), "data file should be an xlsx file"
        FAIL_MSG = "Failed to obtain answer via API."
        tmp_file = eval_file.replace(".xlsx", "_tmp.pkl")
        # tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace(".xlsx", "_score.xlsx")
        score_file_csv = eval_file.replace(".xlsx", "_score.csv")

        if not osp.exists(score_file):

            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            cnt_rejected = 0
            data_un = data[~pd.isna(data["prediction"])]

            for idx in data["index"]:
                ans = data.loc[data["index"] == idx, "answer"].values[0]
                pred = data.loc[data["index"] == idx, "prediction"].values[0]
                output_type = data.loc[data["index"] == idx, "type"].values[0]

                if output_type == "MCQ":
                    extract_pred = pred  # extract_characters_regex(pred)
                    if extract_pred == "":
                        cnt_rejected += 1
                        data.loc[data["index"] == idx, "hit"] = 0
                    else:
                        data.loc[data["index"] == idx, "hit"] = int(extract_pred == ans)
                elif output_type == "YN":
                    extract_pred_yn = self.yn_Extraction(
                        pred[:3]
                    )  # YOrN_Extraction(pred)
                    ans_yn = self.yn_Extraction(ans[:3])
                    if ans_yn == "yes" or ans_yn == "no":
                        ans = ans_yn
                        pred = extract_pred_yn
                    if pred == "Unknown":
                        cnt_rejected += 1
                        data.loc[data["index"] == idx, "hit"] = 0
                    else:
                        data.loc[data["index"] == idx, "hit"] = int(
                            pred.strip().lower() == ans.strip().lower()
                        )
                elif output_type.startswith("Number"):
                    try:
                        extract_pred = eval(str(pred.strip()))
                    except Exception:
                        extract_pred = (
                            -1.0
                        )  # pred.strip()  # self.extract_numbers_from_string(pred, True)
                    # if len(extract_pred) == 0:
                    #     cnt_rejected += 1
                    #     data.loc[data['index'] == idx, 'hit'] = 0
                    #     continue
                    # extract_pred = extract_pred[0]
                    ans = eval(str(ans))
                    if output_type == "Number":
                        data.loc[data["index"] == idx, "hit"] = self.compute_mra(
                            ans, extract_pred
                        )  # data.loc[data['index'] == idx, 'hit'] = 0 #
                    elif output_type == "Number_Int":
                        data.loc[data["index"] == idx, "hit"] = int(extract_pred == ans)
                    else:
                        NotImplementedError(f"Unsupported output type {output_type}.")

            print(
                f"Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, "
                f"failed to obtain the score for another {cnt_rejected} questions. "
                f"Those questions will be counted as 0 score in ALL rating."
            )

            dump(data, score_file)
        data = load(score_file)
        acc = report_acc(data)
        dump(acc, score_file_csv)
        return acc


class SIBenchSingleImage(ImageMCQDataset, ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "MixedOutput"

    NEED_EXTRA_PROMPT_SOURCE = [
        "vstibench",
        "MMSI-Bench",
        "3DSRBench",
        "OmniSpatial",
        "Spatial-MM",
        "SpatialMQA",
        "VSI-Bench",
        "STI-Bench",
        "SpatialEval",
        "SITE-Bench",
        "SPHERE-VLM",
        "SRBench",
        "BLINK",
    ]
    SETTING = [
        "relative_distance",
        "Reach_Prediction",
        # "Object_Shape",
        "Height",
        "Existence",
        "Spatial_Compatibility",
        "Coordinate_Conversion",
        # "Counting",
        # "Route_Planning",
        # "Trajectory_Description",
        "Geometric_Reasoning",
        "Spatial_Imagination",
        # "Object_Size_Estimation",
        "Spatial_Grid",
        "Situational_QA",
        # "Velocity_Acceleration",
        "Maze_Navigation",
        # "Temporal-Appearance_Order",
        # "Camera_Pose",
        "Occlusion",
        # "multi-view_reasoning",
        # "Object_Localization",
        "Spatial_Relation",
        "relative_distance_MINI",
        "Reach_Prediction_MINI",
        # "Object_Shape_MINI",
        "Height_MINI",
        "Existence_MINI",
        "Spatial_Compatibility_MINI",
        "Coordinate_Conversion_MINI",
        # "Counting_MINI",
        # "Route_Planning_MINI",
        # "Trajectory_Description_MINI",
        "Geometric_Reasoning_MINI",
        "Spatial_Imagination_MINI",
        # "Object_Size_Estimation_MINI",
        "Spatial_Grid_MINI",
        "Situational_QA_MINI",
        # "Velocity_Acceleration_MINI",
        "Maze_Navigation_MINI",
        # "Temporal-Appearance_Order_MINI",
        # "Camera_Pose_MINI",
        "Occlusion_MINI",
        # "multi-view_reasoning_MINI",
        # "Object_Localization_MINI",
        "Spatial_Relation_MINI",
        "SIBench_Single_Image_MINI",
        "SIBench_Single_Image",
    ]

    def __init__(self, dataset="MMBench", skip_noimg=True):
        super(SIBenchSingleImage, self).__init__(dataset, skip_noimg)

    @classmethod
    def supported_datasets(cls):
        return cls.SETTING

    def add_extra_prompt(self, prompt, answer_type, data_source):
        if data_source in self.NEED_EXTRA_PROMPT_SOURCE:
            if answer_type == "MCQ":
                prompt += "\nSelect from the given options, answer with letters only."
            elif answer_type == "YN":
                prompt += "\nAnswer with 'Yes' or 'No' only."
            elif answer_type.startswith("Number"):
                prompt += "\nAnswer using a single number and nothing else."
            else:
                raise NotImplementedError(
                    f"Answer type '{answer_type}' is not supported. Supported types are: 'MCQ', 'YN', 'Number'."
                )
        elif data_source is None:
            raise KeyError("Required key 'data_source' is missing.")
        return prompt

    def build_prompt_for_image(self, line, data_base):
        msgs = []
        ROOT = LMUDataRoot()
        if line.get("image_path"):
            tgt_path = toliststr("".join(line["image_path"].split()).split(","))
            for _ in range(len(tgt_path)):
                tgt_path[_] = os.path.join(ROOT, "images", tgt_path[_])
        else:
            raise KeyError("Required key 'image_path' is missing.")

        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]

        question = line["question"]
        prompt = question
        answer_type = line.get("type")
        data_source = line.get("data_source")
        prompt = self.add_extra_prompt(prompt, answer_type, data_source)
        msgs.append(dict(type="text", value=prompt))
        return msgs

    def build_prompt(self, line, video_llm=None, data_base="~/LMUData/images"):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if line.get("input_type") in ["image", "multi-view"]:
            return self.build_prompt_for_image(line=line, data_base=data_base)
        else:
            raise NotImplementedError(
                f"Unrecognized input type: {line.get('input_type')}.\
                                       Just support 'image', 'multi-view' and 'video'."
            )

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import extract_characters_regex, report_acc
        from .utils.yorn import YOrN_Extraction

        nproc = judge_kwargs.pop("nproc", 4)
        model_name = judge_kwargs.get("model", "extract_matching")
        if model_name == "exact_matching":
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn(
                "OPENAI_API_KEY is not set properly, will use exact matching for evaluation"
            )
            model = None

        assert eval_file.endswith(".xlsx"), "data file should be an xlsx file"
        FAIL_MSG = "Failed to obtain answer via API."
        tmp_file = eval_file.replace(".xlsx", "_tmp.pkl")
        # tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace(".xlsx", "_score.xlsx")
        score_file_csv = eval_file.replace(".xlsx", "_score.csv")

        if not osp.exists(score_file):

            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            cnt_rejected = 0
            data_un = data[~pd.isna(data["prediction"])]

            meta = self.data
            meta_q_map = {x: y for x, y in zip(meta["index"], meta["question"])}
            data_map = {x: y for x, y in zip(data["index"], data["question"])}
            for k in data_map:
                assert (
                    k in meta_q_map
                ), f"eval_file should be the same as or a subset of dataset {self.dataset_name}"

                data = sibench_vanilla_eval(
                    model, data, meta, nproc, tmp_file, self.dataset_name
                )

            print(
                f"Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, "
                f"failed to obtain the score for another {cnt_rejected} questions. "
                f"Those questions will be counted as 0 score in ALL rating."
            )

            dump(data, score_file)
        data = load(score_file)
        acc = report_acc(data)
        dump(acc, score_file_csv)
        return acc
