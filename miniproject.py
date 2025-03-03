from flask import Flask, request, render_template_string
import cv2
import pytesseract
from PIL import Image
import os
import tempfile
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB

# Configure logging to ensure output is visible
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Image Upload</title>
</head>
<body>
    <h1>Upload an Image for OCR</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Upload">
    </form>
    {% if text %}
        <h2>Extracted Text:</h2>
        <pre>{{ text }}</pre>
    {% endif %}
</body>
</html>
'''

def preprocess_image(image_path):
    """Preprocess the image for OCR."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def recognize_text(image):
    """Extract text from the preprocessed image using Tesseract."""
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(image, config=custom_config)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        # Generate a unique temporary file path
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, next(tempfile._get_candidate_names()) + '_' + file.filename)

        try:
            # Save the uploaded file
            file.save(temp_path)
            logger.info(f"File saved to {temp_path}")

            # Preprocess the image
            preprocessed_image = preprocess_image(temp_path)
            if preprocessed_image is None:
                logger.error(f"Failed to process image: {temp_path}")
                return 'Error: Could not process the image. The file format might be unsupported.', 400

            # Perform OCR
            text = recognize_text(preprocessed_image)
            logger.info("Text extracted successfully")
            return render_template_string(HTML_TEMPLATE, text=text)

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            return 'Error: An unexpected error occurred during processing.', 500

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"Temporary file {temp_path} removed")
                except Exception as e:
                    logger.error(f"Failed to remove temporary file {temp_path}: {str(e)}")

    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)