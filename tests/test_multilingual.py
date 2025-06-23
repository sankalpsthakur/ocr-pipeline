import sys
import pathlib
from unittest.mock import Mock, patch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pipeline
import config


@patch('pipeline.np')
@patch('pipeline._auto_rotate', return_value=Mock())
@patch('pipeline.preprocess_image', return_value=Mock())
@patch('pipeline.pytesseract')
@patch('pipeline.easyocr')
@patch('pipeline.PaddleOCR')
def test_language_settings(mock_paddle, mock_easyocr, mock_tess_mod, mock_pre, mock_rot, mock_np):
    # Setup mocks
    mock_np.array.return_value = Mock()
    mock_tess_mod.image_to_data.return_value = {'text': [], 'conf': []}
    reader_inst = Mock(readtext=Mock(return_value=[]))
    mock_easyocr.Reader.return_value = reader_inst
    paddle_inst = Mock(ocr=Mock(return_value=[]))
    mock_paddle.return_value = paddle_inst

    # Remove cached readers
    for attr in ('reader',):
        if hasattr(pipeline._easyocr_ocr, attr):
            delattr(pipeline._easyocr_ocr, attr)
        if hasattr(pipeline._paddleocr_ocr, attr):
            delattr(pipeline._paddleocr_ocr, attr)

    with patch.object(pipeline, 'TESSERACT_LANG', 'fra'), \
         patch.object(pipeline, 'EASYOCR_LANG', ['en', 'fr']), \
         patch.object(pipeline, 'PADDLEOCR_LANG', 'fr'), \
         patch.object(pipeline, 'OCR_LANG', None), \
         patch.object(config, 'EASYOCR_LANG', ['en', 'fr']), \
         patch.object(config, 'PADDLEOCR_LANG', 'fr'), \
         patch.object(config, 'OCR_LANG', None):
        img = Mock(mode='RGB')
        pipeline._tesseract_ocr(img)
        pipeline._easyocr_ocr(img)
        pipeline._paddleocr_ocr(img)

    assert mock_tess_mod.image_to_data.call_args.kwargs['lang'] == 'fra'
    mock_easyocr.Reader.assert_called_once_with(['en', 'fr'], gpu=pipeline.EASYOCR_GPU)
    assert mock_paddle.call_args.kwargs['lang'] == 'fr'

    # Test global override
    mock_paddle.reset_mock()
    if hasattr(pipeline._paddleocr_ocr, 'reader'):
        delattr(pipeline._paddleocr_ocr, 'reader')
    with patch.object(pipeline, 'OCR_LANG', 'es'):
        pipeline._paddleocr_ocr(img)
    assert mock_paddle.call_args.kwargs['lang'] == 'es'
