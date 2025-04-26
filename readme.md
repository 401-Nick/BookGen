This thing is EXPENSIVE, use it wisely. ~$0.50 per page

# Visual Story Generator

This Python script leverages OpenAI's GPT models (for text and image generation) to create short visual stories based on user prompts. It takes a story idea and a desired visual style, generates narrative text and detailed illustration descriptions for several pages, and then creates corresponding images for each page, saving them locally.

## Features

*   Generates multi-page story structures (text and illustration descriptions) using a configurable language model (default: `"gpt-4o"`).
*   Creates illustrations for each page using OpenAI's image generation API (default: `"gpt-image-1"`).
*   Accepts user input for the story theme/idea and the desired visual art style.
*   Handles JSON parsing from the language model, including attempts to recover from common formatting quirks (e.g., markdown wrappers).
*   Incorporates basic layout guidance (illustration placement, text placement) into image prompts.
*   Saves generated images as PNG files in a designated output folder.
*   Includes configurable settings (API models, target pages, output folder, API delay).
*   Provides logging for monitoring the generation process and diagnosing errors.
*   Includes basic error handling for common API issues like rate limits, bad requests, content policy violations, and other exceptions.

## Prerequisites

*   Python 3.7+
*   An OpenAI API key with access to the required models (e.g., `gpt-4o` for text, `gpt-image-1` or DALL-E 3 equivalent for images).
*   Sufficient OpenAI API credits/quota for generating text and multiple images.

## Installation & Setup

1.  **Clone or Download:** Clone this repository or download the `visual_story_generator.py` script (assuming that's the filename).
2.  **Navigate:** Open your terminal or command prompt and navigate to the directory where you saved the script.
3.  **Install Dependencies:** Install the necessary Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create `.env` file:** Create a file named `.env` in the *same directory* as the script.

## Configuration

1.  **API Key:** Add your OpenAI API key to the `.env` file you just created. The file content should be:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` with your actual secret API key. **Do not share this key.**

2.  **Script Constants (Optional):** You can modify constants directly within the Python script for further customization:
    *   `STORY_MODEL`: The language model for story generation (default: `"gpt-4o"`).
    *   `IMAGE_MODEL`: The model for image generation (default: `"gpt-image-1"`).
    *   `TARGET_PAGES`: The desired number of pages (default: `4`).
    *   `DEFAULT_ASPECT_RATIO`: Default image aspect ratio (e.g., `"1:1"`, `"16:9"`, `"9:16"`). Affects image dimensions.
    *   `OUTPUT_FOLDER`: Directory to save generated images (default: `"generated_visual_story"`).
    *   `API_DELAY_SECONDS`: Pause between image generation calls to help manage rate limits (default: `10`).

## Usage

1.  **Ensure Setup:** Make sure you have completed the Installation & Setup steps, especially configuring your `.env` file.
2.  **Run Script:** Execute the script from your terminal within the correct directory:
    ```bash
    python app.py
    ```
3.  **Follow Prompts:** A server at 127.0.0.1:5000 will serve as the UI for the application. Fill out the form with:
    *   The core idea or theme for your story (e.g., `"a cat who learns to fly"`, `"cyberpunk detective in Neo-Tokyo"`).
    *   The desired visual theme or art style (e.g., `"watercolor illustration"`, `"pixel art"`, `"photorealistic dark fantasy"`).
    *   The desired number of pages for your story (e.g., `4`).
4.  **Wait for Generation:** The script will first generate the story structure (text and image descriptions) and then proceed to generate an image for each page. Progress and potential errors will be logged to the console.
5.  **Find Output:** Once completed, the generated PNG images (one per page) will be saved in the folder specified by `OUTPUT_FOLDER` (default: `"generated_visual_story"`). WILL ADD FUNCTIONALITY TO SHOW THE GENERATED BOOK ONCE IT IS COMPLETED IN THE UI, BUT FOR NOW JUST LOOK IN THE FOLDER.

## Error Handling & Troubleshooting

*   **Logging:** The script outputs informational messages and errors to the console. Check these messages first if something goes wrong.
*   **API Key:** Ensure your `OPENAI_API_KEY` in the `.env` file is correct and active.
*   **Quota/Credits:** Verify you have sufficient funds or credits in your OpenAI account.
*   **Rate Limits:** If you encounter `RateLimitError`, the script might be making API calls too quickly for your account tier. Try increasing the `API_DELAY_SECONDS` value in the script.
*   **Content Policy:** OpenAI's safety system might block prompts it deems unsafe. If you get content policy errors, try rephrasing your story idea, style, or review the generated `illustration_content` that caused the error (logged by the script).
*   **JSON Errors:** The script attempts to handle common LLM JSON output issues, but complex errors might still occur. Check the logs for details about parsing failures.
*   **Model Access:** Ensure your API key has permission to use the specific models defined in `STORY_MODEL` and `IMAGE_MODEL`.