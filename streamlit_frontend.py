"""
Streamlit frontend for Agentic Churn Predictor (LLM mapping)

- Upload an Excel/CSV file and pick the desired sheet (auto-detected for Excel).
- The UI sends the file and sheet_name to backend POST /predict-file (multipart/form-data).
- After receiving predictions, you can preview, download CSV/Excel, or click 'View Excel' to render the Excel in the UI.
"""

import streamlit as st
import pandas as pd
import io
import requests
import json

st.set_page_config(page_title='Agentic Churn UI (sheet)', layout='wide')
st.title('Agentic Churn Predictor — Upload sheet')

st.sidebar.header('Backend connection')
backend_url = st.sidebar.text_input('Backend URL', 'http://localhost:8000')
request_timeout = st.sidebar.number_input('Request timeout (seconds)', min_value=30, max_value=900, value=180, step=30)
st.sidebar.markdown('The backend must have the model pickle placed locally (or set MODEL_PATH). If you want LLM mapping, set OPENAI_API_KEY on the backend host.')

st.header('1) Upload Excel/CSV and choose sheet')
data_file = st.file_uploader('Upload Excel (.xlsx/.xls) or CSV (.csv) data file', type=['xlsx', 'xls', 'csv'])

# placeholders for sheet picker and file preview
sheet_selector = st.empty()
preview_area = st.empty()
logs_box = st.empty()

# session-state to store generated outputs across reruns
if 'last_predictions_df' not in st.session_state:
    st.session_state['last_predictions_df'] = None
if 'last_csv_bytes' not in st.session_state:
    st.session_state['last_csv_bytes'] = None
if 'last_excel_bytes' not in st.session_state:
    st.session_state['last_excel_bytes'] = None

# helper: get bytes and ext
def _get_file_bytes_and_ext(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
    except Exception:
        # fallback
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
    ext = uploaded_file.name.split('.')[-1].lower()
    return file_bytes, ext

# detect sheet names for Excel and show selectbox
selected_sheet = ""
sheet_options = []
if data_file is not None:
    file_bytes, ext = _get_file_bytes_and_ext(data_file)
    if ext in ('xls', 'xlsx'):
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet_options = xls.sheet_names[:]  # list of sheet names
            # add a friendly choice for first sheet
            sheet_display_options = ['(first sheet)'] + sheet_options
            selected_sheet = sheet_selector.selectbox('Choose sheet', sheet_display_options, index=0)
        except Exception as e:
            logs_box.error(f'Failed to read sheet names: {e}')
            selected_sheet = sheet_selector.text_input('Sheet name or index (leave blank for first sheet)', value='')
    else:
        # CSV -> no sheets
        selected_sheet = sheet_selector.text_input('Sheet name (not used for CSV) — leave blank', value='')
    # show a small preview of the selected sheet
    try:
        if ext in ('xls', 'xlsx'):
            # determine actual sheet to preview
            actual_sheet = None
            if selected_sheet == '(first_sheet)':
                actual_sheet = 0
            elif selected_sheet == '(first sheet)':
                actual_sheet = 0
            elif selected_sheet == '' or selected_sheet is None:
                actual_sheet = 0
            else:
                # if selected is a sheet name use it
                if selected_sheet in sheet_options:
                    actual_sheet = selected_sheet
                else:
                    # try numeric index
                    try:
                        actual_sheet = int(selected_sheet)
                    except Exception:
                        actual_sheet = 0
            preview_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=actual_sheet, nrows=10, engine='openpyxl')
        else:
            data_file.seek(0)
            preview_df = pd.read_csv(io.BytesIO(file_bytes), nrows=10)
        preview_area.subheader('Preview (first 10 rows) of chosen sheet/file')
        preview_area.dataframe(preview_df)
    except Exception as e:
        preview_area.error(f'Preview failed: {e}')
else:
    # no file, show input box so UI doesn't shift too much
    selected_sheet = sheet_selector.text_input('Sheet name or sheet index (leave blank to use first sheet)', value='')

st.write('---')
col1, col2 = st.columns([1,3])
with col1:
    run_btn = st.button('Run prediction (uses server-side model)')
    run_csv_btn = st.button('Run and download CSV from server')
with col2:
    st.write('The backend agent will map sheet columns to the model features (LLM if configured) and return predictions.')

mapping_box = st.empty()
pred_preview_box = st.empty()
download_box = st.empty()

# helper: build bytes for csv & excel and store in session_state
def _store_results(pred_df: pd.DataFrame):
    # CSV
    csv_buf = io.StringIO()
    pred_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode('utf-8')
    st.session_state['last_csv_bytes'] = csv_bytes

    # Excel
    excel_buf = io.BytesIO()
    try:
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            pred_df.to_excel(writer, index=False, sheet_name='predictions')
        excel_buf.seek(0)
        excel_bytes = excel_buf.read()
        st.session_state['last_excel_bytes'] = excel_bytes
    except Exception as e:
        # if Excel write fails, clear excel bytes and show error later
        st.session_state['last_excel_bytes'] = None
        logs_box.warning(f'Failed to generate Excel bytes: {e}')

    st.session_state['last_predictions_df'] = pred_df

# When user clicks Run
if run_btn:
    if data_file is None:
        st.error('Please upload a data file first.')
    else:
        file_bytes, ext = _get_file_bytes_and_ext(data_file)
        # determine requested sheet string to send to backend
        if ext in ('xls', 'xlsx'):
            # map UI selected label to actual sheet identifier string payload
            if selected_sheet == '(first sheet)' or selected_sheet == '':
                sheet_payload = ''
            else:
                sheet_payload = selected_sheet
        else:
            sheet_payload = ''  # CSV doesn't use sheet

        files = {'file': (data_file.name, file_bytes)}
        data = {'sheet_name': sheet_payload}
        predict_endpoint = backend_url.rstrip('/') + '/predict-file'
        with st.spinner('Uploading file and asking the agent (backend)...'):
            try:
                # reduced default timeout since backend now batches/parallelizes
                r = requests.post(predict_endpoint, files=files, data=data, timeout=int(request_timeout))
                if r.status_code != 200:
                    try:
                        err = r.json()
                    except Exception:
                        err = r.text
                    st.error(f'Backend error: {r.status_code} - {err}')
                else:
                    resp = r.json()
                    mapping = resp.get('mapping')
                    mapping_source = resp.get('mapping_source', 'unknown')
                    predictions = resp.get('predictions')

                    mapping_box.subheader(f'Feature mapping used by agent (source: {mapping_source})')
                    mapping_box.json(mapping)

                    if predictions:
                        pred_df = pd.DataFrame(predictions)
                        _store_results(pred_df)

                        pred_preview_box.subheader('Predictions preview (first 10 rows)')
                        pred_preview_box.dataframe(pred_df.head(10))

                        # CSV download
                        if st.session_state['last_csv_bytes'] is not None:
                            download_box.download_button(
                                'Download full results CSV',
                                data=st.session_state['last_csv_bytes'],
                                file_name='predictions.csv',
                                mime='text/csv'
                            )

                        # Excel download + view (if excel bytes created)
                        if st.session_state['last_excel_bytes'] is not None:
                            colA, colB = st.columns([1,1])
                            with colA:
                                st.download_button('Download full results (Excel)', data=st.session_state['last_excel_bytes'],
                                                   file_name='predictions.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                            with colB:
                                if st.button('View Excel (render in browser)'):
                                    try:
                                        excel_buf = io.BytesIO(st.session_state['last_excel_bytes'])
                                        df_from_excel = pd.read_excel(excel_buf, sheet_name=0, engine='openpyxl')
                                        st.subheader('Excel contents (rendered)')
                                        st.dataframe(df_from_excel)
                                    except Exception as e:
                                        st.error(f'Could not render Excel: {e}')
                        else:
                            # fallback: still show CSV as downloadable and render CSV as table
                            if st.session_state['last_csv_bytes'] is not None:
                                # render CSV table
                                try:
                                    csv_buf = io.BytesIO(st.session_state['last_csv_bytes'])
                                    df_csv = pd.read_csv(io.TextIOWrapper(csv_buf, encoding='utf-8'))
                                    st.subheader('Full results (rendered from CSV)')
                                    st.dataframe(df_csv)
                                except Exception:
                                    pass

                    st.success(f'Prediction completed — {resp.get("n_rows", "?")} rows processed')
            except Exception as e:
                st.exception(e)

if run_csv_btn:
    if data_file is None:
        st.error('Please upload a data file first.')
    else:
        file_bytes, ext = _get_file_bytes_and_ext(data_file)
        if ext in ('xls', 'xlsx'):
            if selected_sheet == '(first sheet)' or selected_sheet == '':
                sheet_payload = ''
            else:
                sheet_payload = selected_sheet
        else:
            sheet_payload = ''

        files = {'file': (data_file.name, file_bytes)}
        data = {'sheet_name': sheet_payload}
        predict_endpoint = backend_url.rstrip('/') + '/predict-file-csv'
        with st.spinner('Uploading and running predictions (CSV streaming)...'):
            try:
                r = requests.post(predict_endpoint, files=files, data=data, timeout=int(request_timeout), stream=True)
                if r.status_code != 200:
                    try:
                        err = r.json()
                    except Exception:
                        err = r.text
                    st.error(f'Backend error: {r.status_code} - {err}')
                else:
                    content = r.content
                    st.download_button('Download predictions.csv', data=content, file_name='predictions.csv', mime='text/csv')
                    try:
                        # preview first 2000 bytes
                        preview_df = pd.read_csv(io.StringIO(content.decode('utf-8')), nrows=10)
                        pred_preview_box.subheader('CSV preview (first 10 rows)')
                        pred_preview_box.dataframe(preview_df)
                    except Exception:
                        pass
                    st.success('CSV is ready.')
            except Exception as e:
                st.exception(e)

st.write('---')
st.write('Notes:')
st.write('- Place model pickle on backend host (default ./model.pkl) or set MODEL_PATH env var on the backend host.')
st.write('- For LLM mapping, set OPENAI_API_KEY on the backend host and optionally OPENAI_MODEL.')
