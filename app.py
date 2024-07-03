#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import logging
import uuid
import os
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer
import shutil
import subprocess

# 设置基本路径
base_path = './model'

# 检查目标目录是否存在，如果不存在则克隆仓库
if not os.path.exists(base_path):
    clone_command = f'git clone https://code.openxlab.org.cn/chenshufan/query_preprocess.git {base_path}'
    clone_result = os.system(clone_command)
    if clone_result != 0:
        raise RuntimeError(f"Failed to clone repository with command: {clone_command}")

    # 安装 Git LFS
    lfs_install_command = 'git lfs install'
    print(f"Running LFS install command: {lfs_install_command}")
    lfs_install_result = os.system(lfs_install_command)
    if lfs_install_result != 0:
        raise RuntimeError(f"Failed to install Git LFS with command: {lfs_install_command}")

    # 拉取 LFS 文件
    lfs_pull_command_t5 = f'cd {base_path}/T5-small && git lfs pull'
    lfs_pull_command_flan = f'cd {base_path}/flan-T5-base && git lfs pull'

    try:
        lfs_pull_result_t5 = subprocess.run(lfs_pull_command_t5, shell=True, check=True, capture_output=True)
        print(f"LFS pull output for T5-small: {lfs_pull_result_t5.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"LFS pull error output for T5-small: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to pull LFS files for T5-small with command: {lfs_pull_command_t5}")

    print(f"Running LFS pull command for flan-T5-base: {lfs_pull_command_flan}")
    try:
        lfs_pull_result_flan = subprocess.run(lfs_pull_command_flan, shell=True, check=True, capture_output=True)
        print(f"LFS pull output for flan-T5-base: {lfs_pull_result_flan.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"LFS pull error output for flan-T5-base: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to pull LFS files for flan-T5-base with command: {lfs_pull_command_flan}")

# 设置日志记录器
def setup_logger():
    if 'logger_configured' not in st.session_state:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        log_file_path = './log/st_log.log'
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        file_handler_name = 'streamlit_file_handler'
        streamlit_root_logger = logging.getLogger(st.__name__)
        if not any(handler.get_name() == file_handler_name for handler in streamlit_root_logger.handlers):
            file_handler.set_name(file_handler_name)
            streamlit_root_logger.addHandler(file_handler)
        st.session_state['logger_configured'] = True

setup_logger()
streamlit_root_logger = logging.getLogger(st.__name__)

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())
user_id = st.session_state['user_id']

# 加载模型和tokenizer
t5_path = f"{base_path}/T5-small"
model_t5 = T5ForConditionalGeneration.from_pretrained(t5_path).to(torch.device("cuda"))
tokenizer_t5 = T5Tokenizer.from_pretrained(t5_path)

flan_t5_path = f"{base_path}/flan-T5-base"
model_flan_t5 = T5ForConditionalGeneration.from_pretrained(flan_t5_path).to(torch.device("cuda"))
tokenizer_flan_t5 = T5Tokenizer.from_pretrained(flan_t5_path)

def process_query(query, flag=0):
    """本地处理用户的查询。"""
    query = 'rewrite query:' + query
    try:
        if flag == 0:
            input_ids = tokenizer_t5.encode(query, return_tensors="pt").to("cuda")
            output_ids = model_t5.generate(input_ids, max_length=100, num_beams=5, temperature=0.7, no_repeat_ngram_size=2)
            output_text = tokenizer_t5.decode(output_ids[0], skip_special_tokens=True)
        elif flag == 1:
            input_ids = tokenizer_flan_t5.encode(query, return_tensors="pt").to("cuda")
            output_ids = model_flan_t5.generate(input_ids, max_length=100, num_beams=5, temperature=0.7, no_repeat_ngram_size=2)
            output_text = tokenizer_flan_t5.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    except Exception as e:
        return f"Error: {str(e)}"

# 创建 Streamlit UI 组件
st.title("Query processing module")
user_input = st.text_input("Enter your query to preprocess", placeholder="Type your english question here...")

if 'result' not in st.session_state:
    st.session_state['result'] = None

if 'feedback_given' not in st.session_state:
    st.session_state['feedback_given'] = False

if st.button("Extract keywords"):
    if user_input:
        st.write(f"User ({user_id}): {user_input}")
        streamlit_root_logger.info(f"User ({user_id}) query (Extract keywords): {user_input}")
        try:
            result = process_query(user_input, flag=0)
            st.write(f"Bot: {result}")
            st.session_state['result'] = result
            st.session_state['feedback_given'] = False
            streamlit_root_logger.info(f"Response (Extract keywords) to {user_id} : {result}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            streamlit_root_logger.error(f"Error (Extract keywords): {e}")
    else:
        st.error("Please enter a query to send.")

if st.button("Generate sub-queries"):
    if user_input:
        st.write(f"User ({user_id}): {user_input}")
        streamlit_root_logger.info(f"User ({user_id}) query (Generate sub-queries): {user_input}")
        try:
            result = process_query(user_input, flag=1)
            st.write(f"Bot: {result}")
            st.session_state['result'] = result
            st.session_state['feedback_given'] = False
            streamlit_root_logger.info(f"Response (Generate sub-queries) to {user_id}: {result}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            streamlit_root_logger.error(f"Error (Generate sub-queries): {e}")
    else:
        st.error("Please enter a query to send.")

# 用户反馈功能
if st.session_state['result'] is not None and not st.session_state['feedback_given']:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("满意"):
            streamlit_root_logger.info(f"User satisfaction (满意) for user ({user_id})")
            st.session_state['feedback_given'] = True
            st.rerun()
    with col2:
        if st.button("不满意"):
            streamlit_root_logger.info(f"User satisfaction (不满意) for user ({user_id})")
            st.session_state['feedback_given'] = True
            st.rerun()

if st.session_state['feedback_given']:
    st.write("Thank you for your feedback!")
    st.write(f"Bot: {st.session_state['result']}")
