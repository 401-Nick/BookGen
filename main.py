import os
import base64
import json
import time
import logging
from typing import List, Dict, Optional, Any # Import necessary types
from openai import OpenAI, RateLimitError, APIError, BadRequestError # Import specific exceptions
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI Client (ensure API key is loaded from .env or environment variables)
try:
    client = OpenAI()
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client. Ensure OPENAI_API_KEY is set: {e}")
    exit(1) # Exit if client can't be initialized

# --- Model & Generation Settings ---

# Text Generation Model (for writing the story)
STORY_MODEL = "gpt-4o" 

# Image Generation Model
IMAGE_MODEL = "gpt-image-1"

# Story Constraints
TARGET_PAGES = 4

# Default Page Layout & Style Settings
DEFAULT_ASPECT_RATIO = "1:1" 
DEFAULT_ILLUSTRATION_ORIENTATION = "primarily in the top or center part of the image"
DEFAULT_TEXT_ORIENTATION = "clearly readable, usually placed in the bottom part or an area with clear background"
BASE_STYLE_NOTES = "Ensure character descriptions are consistent across pages if characters reappear."

# Output Settings
OUTPUT_FOLDER = "generated_visual_story"

# Rate Limiting / Delays
API_DELAY_SECONDS = 10

# --- Function to Generate Story Structure ---
def generate_story_structure(user_prompt: str, story_style_prompt: str, target_pages: int) -> Optional[List[Dict[str, str]]]:
    """
    Uses an LLM to generate the story content and structure in JSON format,
    tailored to a specific theme/style. Handles various JSON output quirks from the LLM.

    Args:
        user_prompt: The user's initial idea for the story.
        story_style_prompt: The user's desired theme or art style.
        target_pages: The desired number of pages for the story.

    Returns:
        A list of dictionaries, where each dictionary represents a page
        (with 'illustration_content' and 'text_content'), or None if generation fails.
    """
    logging.info(f"--- Generating Story Structure for: '{user_prompt}' ---")
    logging.info(f"--- Applying Theme/Style: '{story_style_prompt}' ---")
    logging.info(f"Using model: {STORY_MODEL}, Target pages: {target_pages}")

    # If no style is provided, use a neutral default.
    if not story_style_prompt:
        story_style_prompt = "a clear and visually engaging style"
        logging.info("No specific style provided, using default.")

    system_prompt = f"""
    You must generate a JSON object with {target_pages} pages.
You are a creative storyteller and visual director. Your task is to write a short, engaging visual story based on the user's prompt.
The story should reflect the following theme or art style: **{story_style_prompt}**.
Break the story down into approximately {target_pages} distinct pages.
The first page must be a cover page with a title. The last page must be a final page with an ending.

For EACH page, you MUST provide:
1.  `text_content`: The narrative text for that page. Keep it concise and suitable for the overall tone.
2.  `illustration_content`: A detailed visual description for the illustration on that page. This description MUST:
    *   Clearly depict the scene, characters, and action described in the `text_content`.
    *   **Crucially, align with the requested theme/style: {story_style_prompt}.** Describe visual elements (colors, textures, mood, composition) that fit this style.
    *   Mention the main characters by consistent names or descriptions (e.g., "the grizzled detective", "the ethereal forest spirit"). This helps maintain visual consistency.
    *   Focus on visual elements for image generation. Be descriptive but clear.

Output Format:
You MUST output the result ONLY as a valid JSON list of dictionaries. Each dictionary represents one page and MUST contain the keys "illustration_content" and "text_content".
Do not include any introductory text, explanations, markdown formatting (like ```json), or anything outside the JSON list itself.

Example (assuming a 'cyberpunk noir' style prompt):
  {{
    "illustration_content": "Cover image: A lone figure in a trench coat standing under flickering neon signs in a rainy, futuristic city alley. Moody, high contrast lighting. Cyberpunk noir style.",
    "text_content": "(Book Title - e.g., Chrome & Shadow)"
  }},
  {{
    "illustration_content": "Close up on a worn, cybernetic hand placing a data chip onto a grimy bar counter. Dim, smoky lighting reflects off the metal. Detailed textures fitting the cyberpunk noir aesthetic.",
    "text_content": "The job was simple. The data was hot. But in Neo-Kyoto, nothing stays simple for long."
  }},
  {{
    "illustration_content": "A wide shot of a bustling, multi-layered street market filled with holographic ads, strange vendors, and augmented people. Flying vehicles pass overhead through the perpetual twilight. The main character is a small figure navigating the crowd. Cyberpunk noir atmosphere.",
    "text_content": "The market buzzed with a thousand secrets, each one for sale."
  }}

  // ... YOU MUST PROVIDE {target_pages} PAGES
"""

    try:
        response = client.chat.completions.create(
            model=STORY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )


        story_json_string = response.choices[0].message.content
        print("HERE")
        print(story_json_string)
        logging.info("--- Story Model Response Received ---")
        # logging.debug(f"Raw story response: {story_json_string}")
        logging.debug(f"Story JSON String: {story_json_string}")

        if not story_json_string:
             logging.error("Story model returned an empty response.")
             return None

        # Attempt to parse the JSON string
        parsed_data = None
        cleaned_json_string = story_json_string.strip()

        try:
            # First, try standard parsing
            parsed_data = json.loads(cleaned_json_string)

        except json.JSONDecodeError as e:
            logging.error(f"!!! Error: Failed to decode JSON from story model. Details: {e}")
            logging.error(f"!!! Attempted to parse:\n{cleaned_json_string}")
            # If standard parsing fails, check for markdown wrappers
            if cleaned_json_string.startswith("```json") and cleaned_json_string.endswith("```"):
                logging.warning("Detected markdown formatting in JSON response, attempting to strip.")
                cleaned_json_string = cleaned_json_string[7:-3].strip()
                try:
                    parsed_data = json.loads(cleaned_json_string)
                except json.JSONDecodeError as e2:
                    logging.error(f"!!! Error: Failed to decode JSON even after stripping markdown.")
                    logging.error(f"   Details: {e2}")
                    logging.error(f"   Attempted to parse:\n{cleaned_json_string}")
                    logging.error(f"   Original Raw Response:\n{story_json_string}")
                    return None
            # If it wasn't markdown, check for potential list structure hidden in text
            elif '[' in cleaned_json_string and ']' in cleaned_json_string:
                 start_index = cleaned_json_string.find('[')
                 end_index = cleaned_json_string.rfind(']')
                 if start_index != -1 and end_index != -1 and end_index > start_index:
                     logging.warning("JSONDecodeError, but found list-like structure. Attempting to extract list.")
                     list_candidate = cleaned_json_string[start_index : end_index + 1]
                     try:
                         parsed_data = json.loads(list_candidate)
                     except json.JSONDecodeError as e3:
                         logging.error(f"!!! Error: Failed to decode extracted list structure.")
                         logging.error(f"   Details: {e3}")
                         logging.error(f"   Attempted to parse list candidate:\n{list_candidate}")
                         logging.error(f"   Original Raw Response:\n{story_json_string}")
                         return None
            else:
                # Original error if no known recovery pattern matches
                logging.error(f"!!! Error: Failed to decode JSON from story model.")
                logging.error(f"   Details: {e}")
                logging.error(f"   Attempted to parse:\n{cleaned_json_string}")
                logging.error(f"   Original Raw Response:\n{story_json_string}")
                return None

        # --- LOGIC: Check if the parsed data is the expected list, a dict containing the list, or just a single page dict ---
        book_pages_data = None
        if isinstance(parsed_data, list):
            logging.info("JSON structure is a list (expected format).")
            book_pages_data = parsed_data
        elif isinstance(parsed_data, dict):
            # --- NEW CHECK: Is the dictionary *itself* a single page? ---
            if "illustration_content" in parsed_data and "text_content" in parsed_data:
                logging.warning("JSON structure is a single dictionary object that looks like a page. Wrapping it in a list.")
                print("Single page dictionary detected.")
                print(parsed_data)
                logging.debug(f"Single page dictionary: {parsed_data}")
                book_pages_data = [parsed_data] # Treat it as a list containing one page
            else:
                # --- OLD CHECK: Does the dictionary contain page objects as values (e.g., {"0": {...}, "1": {...}})? ---
                logging.warning("JSON structure is a dictionary, but not a single page. Attempting to extract page list from values.")
                potential_list = list(parsed_data.values())
                # Basic check: Are the values themselves dictionaries with the required keys?
                if potential_list and all(isinstance(item, dict) and "illustration_content" in item and "text_content" in item for item in potential_list):
                    book_pages_data = potential_list
                    logging.info(f"Successfully extracted {len(book_pages_data)} pages from dictionary values.")
                else:
                    logging.error("JSON dictionary values do not appear to be page dictionaries or are missing required keys.")
                    logging.error(f"Received dictionary keys: {list(parsed_data.keys())}")
                    # Log first few values if they exist, otherwise the whole dict
                    if potential_list:
                         logging.error(f"First few dictionary values: {potential_list[:3]}")
                    else:
                        logging.error(f"Problematic dictionary structure (empty values?): {parsed_data}")
                    return None
        else:
             logging.error(f"Parsed JSON is neither a list nor a recognized dictionary structure. Type: {type(parsed_data)}")
             logging.error(f"Received structure: {parsed_data}")
             return None

        # --- Validation (applies to book_pages_data however it was derived) ---
        if book_pages_data is None:
             # This case should ideally be caught above, but acts as a failsafe
             logging.error("Failed to derive a valid page list structure from the model response.")
             return None

        if not book_pages_data: # Handles empty list derived from response
            logging.warning("Story model resulted in an empty list of pages.")
            return [] # Return an empty list to avoid downstream errors

        # Validate required keys in each page dictionary within the final list
        invalid_pages = []
        for i, page in enumerate(book_pages_data):
            if not isinstance(page, dict) or "illustration_content" not in page or "text_content" not in page:
                invalid_pages.append(f"Page index {i}: {page}")

        if invalid_pages:
            logging.error(f"One or more page items in the final list are invalid or missing keys ('illustration_content', 'text_content').")
            logging.error(f"Invalid items found: {invalid_pages}")
            return None

        logging.info(f"--- Story JSON Parsed Successfully ({len(book_pages_data)} pages) ---")
        return book_pages_data

    except RateLimitError as e:
         logging.error(f"!!! Error: OpenAI API rate limit exceeded during story generation: {e}. Try increasing API_DELAY_SECONDS or check your plan limits.")
         return None
    except APIError as e:
         logging.error(f"!!! Error: OpenAI API error during story generation: {e}")
         return None
    except BadRequestError as e:
        logging.error(f"!!! Error: Bad request to OpenAI API during story generation (check model name, parameters): {e}")
        return None
    except Exception as e:
        logging.error(f"!!! Unexpected error generating story structure: {e}")
        import traceback # Import traceback for detailed error logging
        logging.error(traceback.format_exc()) # Log full stack trace
        return None

# --- Helper Function to Generate a Single Page Image ---
def generate_page_image(page_data: Dict[str, str], page_number: int, requested_style: str) -> bool:
    """
    Generates and saves a single page image using DALL-E based on the requested style.

    Args:
        page_data: Dictionary containing 'illustration_content' and 'text_content'.
        page_number: The sequential number of the page.
        requested_style: The primary style instruction for the image.

    Returns:
        True if image generation and saving were successful, False otherwise.
    """
    logging.info(f"--- Generating Image for Page {page_number} ---")

    # Get settings for this page, using defaults if not specified
    aspect_ratio = page_data.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    illustration_orientation = page_data.get("illustration_orientation", DEFAULT_ILLUSTRATION_ORIENTATION)
    text_orientation = page_data.get("text_orientation", DEFAULT_TEXT_ORIENTATION)
    illustration_content = page_data.get("illustration_content", "No illustration description provided.")
    # Handle potentially missing text content gracefully in the prompt
    text_content = page_data.get("text_content")
    if not text_content: # Check if it's None or empty string
         logging.warning(f"Page {page_number} has missing 'text_content'. Generating image without text.")
         text_prompt_part = "No text should be included on the image."
    else:
        # Limit text length for prompt sanity, DALL-E may truncate anyway
        text_content_limited = text_content[:250]
        text_prompt_part = f'Text to include: Incorporate the following text clearly and legibly onto the image: "{text_content_limited}". Ideally, place the text {text_orientation}. Ensure correct spelling.'


    image_model = page_data.get("image_model", IMAGE_MODEL) # Allow overriding model per page

    # Determine image size based on aspect ratio for DALL-E 3
    if aspect_ratio == "16:9":
        size = "1792x1024"
    elif aspect_ratio == "1:1":
        size = "1024x1024"
    elif aspect_ratio == "9:16":
        size = "1024x1792"
    else:
        logging.warning(f"Unsupported aspect ratio '{aspect_ratio}' for page {page_number}. Using default {DEFAULT_ASPECT_RATIO} (1024x1792).")
        size = "1024x1792" # Fallback to default

    # Construct the DALL-E prompt, leading with the style
    full_style_prompt = f"{requested_style}. {BASE_STYLE_NOTES}"

    prompt = f"""Create an illustration in the style of: **{full_style_prompt}**.
Image Aspect Ratio: {aspect_ratio}.
Scene Description: {illustration_content}. Focus the main illustration elements {illustration_orientation}.
{text_prompt_part}
Visual Style Guidance: Strictly adhere to the requested style: "{requested_style}". Maintain consistent character appearance if characters are described as reappearing across pages.
"""

    logging.info(f"Image Prompt (Page {page_number}): Style='{requested_style}', Scene='{illustration_content[:100]}...' Text='{str(text_content)[:50]}...'") # Log key parts

    try:
        generation_start_time = time.time()
        result = client.images.generate(
            model=image_model,
            prompt=prompt,
            n=1,
            size=size,
        )
        generation_time = time.time() - generation_start_time
        logging.info(f"Image generated in {generation_time:.2f} seconds.")


        image_base64 = result.data[0].b64_json
        if not image_base64:
             logging.error(f"Image generation for page {page_number} returned empty data.")
             return False

        image_bytes = base64.b64decode(image_base64)

        # Ensure output directory exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Save the image
        filename = os.path.join(OUTPUT_FOLDER, f"page_{page_number:02d}.png")
        with open(filename, "wb") as f:
            f.write(image_bytes)
        logging.info(f"Successfully saved page {page_number} image to {filename}")
        return True

    except RateLimitError as e:
         logging.error(f"!!! Error: OpenAI API rate limit exceeded during image generation for page {page_number}: {e}. Consider increasing API_DELAY_SECONDS.")
         return False
    except APIError as e:
         # Use getattr to safely access response details if available
         response_details = getattr(e, 'response', None)
         status_code = getattr(response_details, 'status_code', 'N/A')
         error_body = getattr(e, 'body', {})
         error_message = error_body.get('message', str(e)) if isinstance(error_body, dict) else str(e)

         if "content_policy_violation" in error_message.lower():
              logging.error(f"!!! Error: Content policy violation detected for page {page_number} (Status: {status_code}).")
              logging.error(f"   Potentially problematic content: Style='{requested_style}', Illustration='{illustration_content}', Text='{text_content}'")
              # logging.error(f"   Full Prompt Sent: {prompt}") # Uncomment cautiously for debugging policy issues
         else:
              logging.error(f"!!! Error: OpenAI API error during image generation for page {page_number} (Status: {status_code}): {error_message}")
         return False
    except BadRequestError as e:
        response_details = getattr(e, 'response', None)
        status_code = getattr(response_details, 'status_code', 'N/A')
        error_body = getattr(e, 'body', {})
        error_message = error_body.get('message', str(e)) if isinstance(error_body, dict) else str(e)
        logging.error(f"!!! Error: Bad request to OpenAI API during image generation for page {page_number} (Status: {status_code}). Check parameters.")
        logging.error(f"   Error Details: {error_message}")
        logging.error(f"   Model: {image_model}, Size: {size}, Prompt Snippet: {prompt[:300]}...")
        return False
    except Exception as e:
        logging.error(f"!!! Unexpected error generating page image {page_number}: {e}")
        import traceback
        logging.error(traceback.format_exc()) # Log full stack trace
        return False

# --- Main Execution ---
def main():
    """ Main function to drive the visual story generation process. """
    logging.info("--- Starting Visual Story Generator ---")

    # 1. Get user input
    story_idea = input("Enter the core idea or theme for your story (e.g., 'a space pirate searching for lost treasure', 'two robots falling in love in a junkyard'): ")
    if not story_idea:
        logging.error("No story idea provided. Exiting.")
        return

    user_style = input("Enter the desired visual theme or art style (e.g., 'Studio Ghibli anime', 'dark fantasy oil painting', 'pixel art', 'steampunk concept art', 'watercolor sketch'): ")
    if not user_style:
        logging.warning("No specific style provided. Illustrations will use a default descriptive style.")
        user_style = "clear, detailed illustration style" # Provide a basic fallback

    # 2. Generate the story structure (JSON)
    book_pages = generate_story_structure(story_idea, user_style, TARGET_PAGES)

    # 3. Validate story structure generation
    if book_pages is None:
        # generate_story_structure now logs detailed errors internally
        logging.error("Could not generate story structure due to errors (see logs above). Exiting.")
        return
    if not book_pages: # Checks for empty list specifically
        logging.warning("Story generation resulted in zero pages. Exiting.")
        return

    # 4. Generate images for each page
    logging.info(f"\n--- Starting Image Generation for {len(book_pages)} Pages ---")
    successful_pages = 0
    total_pages = len(book_pages)

    for i, page_info in enumerate(book_pages):
        page_num = i + 1 # Use 1-based indexing for user-facing logs
        logging.info(f"Processing Page {page_num}/{total_pages}...")

        # Basic check before passing to image generation
        if not isinstance(page_info, dict) or "illustration_content" not in page_info:
             # Allow missing text_content, generate_page_image handles it
             logging.warning(f"Skipping page {page_num} due to invalid data format or missing 'illustration_content': {page_info}")
             continue # Skip this page

        # Pass the user's requested style to the image generator
        if generate_page_image(page_info, page_num, user_style):
            successful_pages += 1
        else:
            logging.warning(f"Failed to generate image for page {page_num}.")
            # Optional: Add logic here to stop on first error if desired
            # stop_on_error = False
            # if stop_on_error:
            #    logging.error("Stopping image generation due to error.")
            #    break

        # Add delay between image generations if not the last page
        if i < total_pages - 1:
             logging.info(f"Waiting {API_DELAY_SECONDS}s before next page...")
             time.sleep(API_DELAY_SECONDS)


    # 5. Completion Summary
    logging.info("\n--- Visual Story Generation Process Complete ---")
    logging.info(f"Successfully generated images for {successful_pages} out of {total_pages} pages.")
    if successful_pages > 0:
        # Use abspath to show the full path clearly
        logging.info(f"Generated pages saved in folder: '{os.path.abspath(OUTPUT_FOLDER)}'")
    if successful_pages < total_pages:
        logging.warning("Some page images could not be generated. Check logs above for specific errors.")
    elif total_pages > 0: # Only print "All successful" if there were pages to generate
        logging.info("All page images generated successfully!")
    else: # Handle case where book_pages was initially empty
         logging.info("No pages were processed.")


if __name__ == "__main__":
    main()