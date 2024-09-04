import sys
import json
import logging
from pathlib import Path
from datetime import datetime as dt


def save_json(json_obj, json_path, append_if_exists=False,
              overwrite_if_exists=False, unique_fn_if_exists=True):
    """Saves a json file

    Arguments:
        json_obj: json, json object
        json_path: Path, path including the file name where the json object
            should be saved to
        append_if_exists: bool, append to the existing json file with the same
            name if it exists (keep the json structure intact)
        overwrite_if_exists: bool, xor with append, overwrites any existing
            target file
        unique_fn_if_exsists: bool, appends the current date and time to the
            file name if the target file exists already.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = json_path.parents[0] / f'{str(json_path.stem)}_{time}' \
                                               f'{str(json_path.suffix)}'

    # 如果要改写
    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, 'w+') as fout:
            json.dump(json_obj, fout, indent=2)  # 这一行调用json.dump函数，将Python对象json_obj序列化成JSON格式，并写入之前打开的
            # 文件（即fout）。参数indent=2意味着输出的JSON数据将以两个空格的缩进格式化，使其更易于阅读。
        return

    # 如果继续追加内容
    if append_if_exists:
        if json_path.exists():
            with open(json_path, 'r') as fin:
                read_file = json.load(fin)
            read_file.update(json_obj)  # read_file.update(json_obj)：这行代码执行了update方法，将json_obj（一个Python字典）中的项
            # 目更新到read_file字典中。如果json_obj中的键在read_file中已经存在，它们的值将被json_obj中的相应值覆盖。如果json_obj中的键在read_file中不存在，这些键值对将被添加到read_file中。
            with open(json_path, 'w+') as fout:
                json.dump(read_file, fout, indent=2)
            return

    with open(json_path, 'w+') as fout:
        json.dump(json_obj, fout, indent=2)


def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step - 1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def init_logging(filename=None):
    """Initialises log/stdout output

    Arguments:
        filename: str, a filename can be set to output the log information to
            a file instead of stdout"""
    log_lvl = logging.INFO
    log_format = '%(asctime)s: %(message)s'
    if filename:
        logging.basicConfig(handlers=[logging.FileHandler(filename),
                                      logging.StreamHandler(sys.stdout)],
                            level=log_lvl,
                            format=log_format)
    else:
        logging.basicConfig(stream=sys.stdout, level=log_lvl,
                            format=log_format)


def get_default_config():
    """Returns a default config file"""
    config = {
        'output_sentence_dir' :r'D:\OneDrive - The University of Liverpool\LLMs\gpt2-translator-pytorch\output_sentence\faiss',
        # folder name to which the result json files are written
        'seed': 42,  # random seed for numpy, random, pytorch
        'gpu': 1,  # 1 calculate on GPU
        'dataset': 'Emotion',
        'num_classes': 2,
        'test_sample_num': 50,
        # Default = False, number of samples per class starting from the test_sample_start_per_class to calculate the influence function for. E.g. if your dataset has 10 classes and you set this value to 1, then the influence functions will be calculated for 10 * 1 test samples, one per class. If False, calculates the influence for all images.
        'test_start_index': 0,
        'recursion_depth': 5000,
        # recursion depth for the s_test calculation. Greater recursion depth improves precision.
        'r_averaging': 1,
        # number of s_test calculations to take the average of. Greater r averaging improves precision.
        'scale': 1000,  # scaling factor during s_test calculation.
        'damp': 0.01,
        'calc_method': 'img_wise',
        'log_filename': None,  # Default None, if set the output will be logged to this file in addition to stdout.
        'logdir' : "\OneDrive - The University of Liverpool\LLMs\gpt2-translator-pytorch\logfile.log",
        'outdir' : 'D:\OneDrive - The University of Liverpool\LLMs\gpt2-translator-pytorch\outdir',
        'TRAIN_DIR': r'D:\OneDrive - The University of Liverpool\LLMs\gpt2-translator-pytorch\data\processed_itt_sub.csv',
        'TEST_DIR': r'D:\OneDrive - The University of Liverpool\LLMs\gpt2-translator-pytorch\data\test_50_faiss.csv'
    }

    return config
