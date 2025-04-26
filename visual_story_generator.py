import os
import base64
import json
import time
import logging
from typing import List, Dict, Optional, Any
from openai import OpenAI, RateLimitError, APIError, BadRequestError
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
    exit(1)

# --- Model & Generation Settings ---

# Text Generation Model (for writing the story)
STORY_MODEL = "gpt-4.1" 

# Image Generation Model
# Updated default based on OpenAI deprecation/model names
IMAGE_MODEL = "gpt-image-1" # Use gpt-image-1

# Story Constraints
# TARGET_PAGES is removed as it will be passed as an argument now

# Default Page Layout & Style Settings
DEFAULT_ASPECT_RATIO = "1:1" 
DEFAULT_ILLUSTRATION_ORIENTATION = "primarily in the top or center part of the image"
DEFAULT_TEXT_ORIENTATION = "clearly readable, usually placed in the bottom part or an area with clear background"
BASE_STYLE_NOTES = "Ensure character descriptions are consistent across pages if characters reappear."
DEFAULT_FONT = "Comic Sans MS" # Using a more standard name, though exact font control is limited
# Output Settings
OUTPUT_FOLDER = "generated_visual_story"

# Rate Limiting / Delays
API_DELAY_SECONDS = 2 # 

# --- Function to Generate Story Structure ---
def generate_story_structure(user_prompt: str, story_style_prompt: str, user_pages: int) -> Optional[List[Dict[str, str]]]:
    """
    Uses an LLM to generate the story content and structure in JSON format,
    tailored to a specific theme/style. Handles various JSON output quirks from the LLM.

    Args:
        user_prompt: The user's initial idea for the story.
        story_style_prompt: The user's desired theme or art style.
        user_pages: The desired number of pages for the story.

    Returns:
        A list of dictionaries, where each dictionary represents a page
        (with 'illustration_content' and 'text_content'), or None if generation fails.
    """
    logging.info(f"--- Generating Story Structure for: '{user_prompt}' ---")
    logging.info(f"--- Applying Theme/Style: '{story_style_prompt}' ---")
    logging.info(f"Using model: {STORY_MODEL}, Target pages: {user_pages}")

    if not story_style_prompt:
        story_style_prompt = "a clear and visually engaging style"
        logging.info("No specific style provided, using default.")

    system_prompt = f"""
    When describing characters, always visually describe them in a way that is consistent throughout the story even across multiple story pages.

    Make sure you abide by content policy and provide images that are safe to generate. You can be gritty, dark, or edgy, but refrain from overly explicit or violent content.
    You must generate a JSON object representing a visual story with exactly {user_pages} pages.
    You are a creative storyteller and visual director. Your task is to write a short, engaging visual story based on the user's prompt.
    The story should reflect the following theme or art style: **{story_style_prompt}**.
    Break the story down into exactly {user_pages} distinct pages.
    The first page MUST be a cover page with a title. The last page MUST be a final page with an ending (e.g., 'The End').

For EACH page, you MUST provide:
1.  `text_content`: The narrative text for that page. Keep it concise and suitable for the overall tone. For the cover page, this should be the book title. For the last page, it could be 'The End' or a concluding phrase.
2.  `illustration_content`: A detailed visual description for the illustration on that page. This description MUST:
    *   Clearly depict the scene, characters, and action described in the `text_content`.
    *   **Crucially, align with the requested theme/style: {story_style_prompt}.** Describe visual elements (colors, textures, mood, composition) that fit this style.
    *   Mention the main characters by consistent names or descriptions (e.g., "the grizzled detective", "the ethereal forest spirit"). This helps maintain visual consistency.
    *   Focus on visual elements for image generation. Be descriptive but clear.

Output Format:
You MUST output the result ONLY as a valid JSON list of dictionaries. Each dictionary represents one page and MUST contain the keys "illustration_content" and "text_content".
Do not include any introductory text, explanations, markdown formatting (like ```json), or anything outside the JSON list itself.

Example (assuming a 'cyberpunk noir' style prompt and user_pages = 3):
[
  {{
    "illustration_content": "Cover image: A lone figure in a trench coat standing under flickering neon signs in a rainy, futuristic city alley. Moody, high contrast lighting. Cyberpunk noir style.",
    "text_content": "Chrome & Shadow"
  }},
  {{
    "illustration_content": "Close up on a worn, cybernetic hand placing a data chip onto a grimy bar counter. Dim, smoky lighting reflects off the metal. Detailed textures fitting the cyberpunk noir aesthetic.",
    "text_content": "The job was simple. The data was hot. But in Neo-Kyoto, nothing stays simple for long."
  }},
  {{
    "illustration_content": "The figure looks back over their shoulder towards the viewer as they disappear into the rain-slicked alley. Only the glowing collar of their coat is visible. Cyberpunk noir atmosphere.",
    "text_content": "The End."
  }}
]
"""

    try:
        logging.info("--- CALLING STORY MODEL ---")
        response = client.chat.completions.create(
            model=STORY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        logging.info("--- STORY MODEL RESPONSE RECEIVED ---")


        story_json_string = response.choices[0].message.content
        logging.debug(f"Raw Story JSON String: {story_json_string}")

        if not story_json_string:
             logging.error("Story model returned an empty response.")
             return None

        parsed_data = None
        try:
            # Attempt to parse the JSON string directly
            parsed_data = json.loads(story_json_string)
            # Sometimes the model wraps the list in a root key (e.g., {"pages": [...]})
            if isinstance(parsed_data, dict) and len(parsed_data) == 1:
                potential_list = next(iter(parsed_data.values()), None)
                if isinstance(potential_list, list):
                    logging.info("Detected JSON dictionary wrapper, extracting list.")
                    parsed_data = potential_list

        except json.JSONDecodeError as e:
            logging.warning(f"Initial JSONDecodeError: {e}. Attempting cleanup...")
            # Fallback: try cleaning potential markdown fences or extracting list manually
            cleaned_json_string = story_json_string.strip()
            if cleaned_json_string.startswith("```json") and cleaned_json_string.endswith("```"):
                logging.warning("Detected markdown formatting, stripping.")
                cleaned_json_string = cleaned_json_string[7:-3].strip()
            elif cleaned_json_string.startswith("```") and cleaned_json_string.endswith("```"):
                 logging.warning("Detected generic markdown formatting, stripping.")
                 cleaned_json_string = cleaned_json_string[3:-3].strip()

            # Try parsing again after potential cleaning
            try:
                parsed_data = json.loads(cleaned_json_string)
                if isinstance(parsed_data, dict) and len(parsed_data) == 1:
                    potential_list = next(iter(parsed_data.values()), None)
                    if isinstance(potential_list, list):
                        logging.info("Detected JSON dictionary wrapper post-cleanup, extracting list.")
                        parsed_data = potential_list

            except json.JSONDecodeError as e2:
                logging.error(f"!!! Error: Failed to decode JSON even after cleanup attempts. Details: {e2}")
                logging.error(f"   Attempted to parse:\n{cleaned_json_string}")
                logging.error(f"   Original Raw Response:\n{story_json_string}")
                # Last resort: try finding the outermost list
                start_index = cleaned_json_string.find('[')
                end_index = cleaned_json_string.rfind(']')
                if start_index != -1 and end_index != -1 and end_index > start_index:
                     logging.warning("Attempting to extract list structure via string slicing.")
                     list_candidate = cleaned_json_string[start_index : end_index + 1]
                     try:
                         parsed_data = json.loads(list_candidate)
                     except json.JSONDecodeError as e3:
                         logging.error(f"!!! Error: Failed to decode extracted list structure. Details: {e3}")
                         logging.error(f"   Attempted to parse list candidate:\n{list_candidate}")
                         return None
                else:
                    logging.error("!!! Error: Could not find list structure in the response.")
                    return None

        book_pages_data = None
        if isinstance(parsed_data, list):
            logging.info("Parsed JSON is a list (expected format).")
            book_pages_data = parsed_data
        # Check moved to initial parsing block
        # elif isinstance(parsed_data, dict):
        #     # Check if it's a wrapper dictionary
        #     if len(parsed_data) == 1:
        #          potential_list = next(iter(parsed_data.values()))
        #          if isinstance(potential_list, list):
        #               logging.warning("JSON structure was a dictionary wrapping a list. Extracting list.")
        #               book_pages_data = potential_list
        #     # Check if it looks like a single page (less likely with the prompt asking for a list)
        #     elif "illustration_content" in parsed_data and "text_content" in parsed_data:
        #         logging.warning("JSON structure is a single dictionary object that looks like a page. Wrapping it in a list.")
        #         book_pages_data = [parsed_data]

        if book_pages_data is None:
             logging.error(f"Parsed JSON is not a list or recognized wrapped list structure. Type: {type(parsed_data)}")
             logging.error(f"Received structure: {parsed_data}")
             return None

        if not book_pages_data:
            logging.warning("Story model resulted in an empty list of pages.")
            return [] # Return empty list, not None, as generation technically succeeded but yielded no pages

        invalid_pages = []
        validated_pages = []
        for i, page in enumerate(book_pages_data):
            if isinstance(page, dict) and "illustration_content" in page and "text_content" in page:
                 # Basic validation passed
                 validated_pages.append(page)
            else:
                invalid_pages.append(f"Page index {i}: {page}")

        if invalid_pages:
            logging.error(f"One or more page items in the list are invalid or missing keys ('illustration_content', 'text_content').")
            logging.error(f"Invalid items found: {invalid_pages}")
            # Decide: return only valid pages or fail completely? Let's return valid ones for now.
            logging.warning(f"Proceeding with {len(validated_pages)} valid pages.")
            # return None # Uncomment this line to fail entirely if any page is invalid

        if not validated_pages:
             logging.error("No valid pages found after validation.")
             return None

        logging.info(f"--- Story JSON Parsed & Validated Successfully ({len(validated_pages)} pages) ---")
        return validated_pages

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
        import traceback
        logging.error(traceback.format_exc())
        return None


# --- Helper Function to Generate a Single Page Image ---
def generate_page_image(page_data: Dict[str, str], page_number: int, requested_style: str) -> bool:
    """
    Generates and saves a single page image based on the requested style.

    Args:
        page_data: Dictionary containing 'illustration_content' and 'text_content'.
        page_number: The sequential number of the page.
        requested_style: The primary style instruction for the image.

    Returns:
        True if image generation and saving were successful, False otherwise.
    """
    logging.info(f"--- Generating Image for Page {page_number} ---")

    # Extract details - use defaults if missing in page_data (less likely now)
    aspect_ratio = page_data.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    illustration_orientation = page_data.get("illustration_orientation", DEFAULT_ILLUSTRATION_ORIENTATION)
    text_orientation = page_data.get("text_orientation", DEFAULT_TEXT_ORIENTATION)
    illustration_content = page_data.get("illustration_content", "No illustration description provided.")
    text_content = page_data.get("text_content", "") # Default to empty string if missing

    # Limit text length for the prompt and determine text part
    text_content_limited = text_content[:250].strip() if text_content else ""
    if not text_content_limited:
         logging.warning(f"Page {page_number} has missing or empty 'text_content'. Generating image without embedded text.")
         text_prompt_part = "No text should be rendered onto the image itself."
    else:
        text_prompt_part = (
            f'Render the following text clearly and legibly onto the image: "{text_content_limited}". '
            f'Place the text {text_orientation}. Ensure correct spelling and that the text is not cut off. '
            f'Use a clear, readable font (ignore specific font family requests like {DEFAULT_FONT}, prioritize legibility).'
        )

    image_model = page_data.get("image_model", IMAGE_MODEL)

    # DALL-E 3 sizes
    if aspect_ratio == "16:9":
        size = "1792x1024"
    elif aspect_ratio == "1:1":
        size = "1024x1024"
    elif aspect_ratio == "9:16":
        size = "1024x1792"
    else:
        logging.warning(f"Unsupported aspect ratio '{aspect_ratio}' for page {page_number}. Using default {DEFAULT_ASPECT_RATIO} (1024x1024).")
        size = "1024x1024"
        aspect_ratio = "1:1" # Update aspect ratio used in prompt

    full_style_prompt = f"{requested_style}. {BASE_STYLE_NOTES}"
    prompt = f"""
Create an illustration with the following characteristics:
Style: **{full_style_prompt}**. Strictly adhere to this visual style.
Scene Description: {illustration_content}. Focus the main illustration elements {illustration_orientation}. Maintain consistent character appearance if characters are described as reappearing across pages.
Aspect Ratio: {aspect_ratio}.
{text_prompt_part}
"""

    logging.info(f"Image Prompt (Page {page_number}): Style='{requested_style}', Scene='{illustration_content[:100]}...' Text='{text_content_limited[:50]}...'")

    try:
        generation_start_time = time.time()
        response = client.images.generate(
            model=image_model,
            prompt=prompt,
            n=1,
            size=size,
            quality="high", # or "hd"
        )
        generation_time = time.time() - generation_start_time
        logging.info(f"Image generated in {generation_time:.2f} seconds.")

        # Accessing the base64 data
        image_base64 = response.data[0].b64_json
        if not image_base64:
             logging.error(f"Image generation for page {page_number} returned empty image data.")
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
    except APIError as e: # More specific error handling if needed
         # Attempt to parse the error details if possible
         response_details = getattr(e, 'response', None)
         status_code = getattr(response_details, 'status_code', 'N/A')
         try:
             error_body = e.response.json() if response_details else getattr(e, 'body', {})
             error_message = error_body.get('error', {}).get('message', str(e))
         except Exception: # Fallback if parsing fails
             error_message = str(e)

         if "content_policy_violation" in error_message.lower() or (isinstance(status_code, int) and status_code == 400 and "content policy" in error_message.lower()):
              logging.error(f"!!! Error: Content policy violation detected for page {page_number} (Status: {status_code}). Skipping page.")
              logging.error(f"   Violating Prompt Snippet: Style='{requested_style}', Illustration='{illustration_content[:150]}...', Text='{text_content_limited[:50]}...'")
         elif isinstance(status_code, int) and status_code == 400 and "billing" in error_message.lower():
             logging.error(f"!!! Error: Billing issue detected (Status: {status_code}). Check your OpenAI account/quota. {error_message}")
         else:
              logging.error(f"!!! Error: OpenAI API error during image generation for page {page_number} (Status: {status_code}): {error_message}")
         return False
    except BadRequestError as e: # Often includes prompt issues or parameter errors
        response_details = getattr(e, 'response', None)
        status_code = getattr(response_details, 'status_code', 'N/A')
        try:
             error_body = e.response.json() if response_details else getattr(e, 'body', {})
             error_message = error_body.get('error', {}).get('message', str(e))
        except Exception:
             error_message = str(e)
        logging.error(f"!!! Error: Bad request to OpenAI API during image generation for page {page_number} (Status: {status_code}). Check parameters/prompt.")
        logging.error(f"   Error Details: {error_message}")
        logging.error(f"   Model: {image_model}, Size: {size}, Prompt Snippet: {prompt[:300]}...")
        return False
    except Exception as e:
        logging.error(f"!!! Unexpected error generating page image {page_number}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

# --- New Function to Generate the Book ---
def generate_book(user_prompt: str, user_style: str, user_pages: int) -> bool:
    """
    Generates a visual story based on user input, handling structure and image generation.

    Args:
        user_prompt: The core idea or theme for the story.
        user_style: The desired visual theme or art style.
        user_pages: The desired number of pages for the story.

    Returns:
        True if the process completed (even with some page failures),
        False if critical errors occurred early (e.g., story structure failure).
    """
    logging.info(f"--- Starting Visual Story Generation for: '{user_prompt}' ---")
    logging.info(f"--- Style: '{user_style}', Target Pages: {user_pages} ---")

    # Handle potential empty style input
    if not user_style:
        logging.warning("No specific style provided. Illustrations will use a default descriptive style.")
        effective_style = "clear, detailed illustration style" # Use a default
    else:
        effective_style = user_style

    # Validate user_pages
    if not isinstance(user_pages, int) or user_pages <= 0:
        logging.error(f"Invalid user_pages value: {user_pages}. Must be a positive integer.")
        return False

    # 1. Generate the story structure (JSON)
    book_pages = generate_story_structure(user_prompt, effective_style, user_pages)

    # 2. Validate story structure generation
    if book_pages is None:
        logging.error("Could not generate story structure due to errors (see logs above). Book generation aborted.")
        return False # Critical failure
    if not book_pages:
        logging.warning("Story generation resulted in zero pages. Nothing to generate images for.")
        return True # Technically didn't fail, just produced nothing

    # 3. Generate images for each page
    logging.info(f"\n--- Starting Image Generation for {len(book_pages)} Pages ---")
    successful_pages = 0
    total_pages_to_generate = len(book_pages) # Use the actual number of pages returned

    # Adjust total expected pages if validation removed some
    if len(book_pages) != user_pages:
        logging.warning(f"Story structure generated {len(book_pages)} pages, which differs from the target of {user_pages}. Proceeding with actual count.")


    for i, page_info in enumerate(book_pages):
        page_num = i + 1
        logging.info(f"Processing Page {page_num}/{total_pages_to_generate}...")

        # Basic check (already done in generate_story_structure, but belt-and-suspenders)
        if not isinstance(page_info, dict) or "illustration_content" not in page_info:
            logging.warning(f"Skipping page {page_num} due to invalid data format or missing 'illustration_content': {page_info}")
            continue

        # Pass the *effective* style used for structure generation
        if generate_page_image(page_info, page_num, effective_style):
            successful_pages += 1
        else:
            logging.warning(f"Failed to generate image for page {page_num}.") # Specific errors logged in generate_page_image

        # Apply delay between image generation calls
        if i < total_pages_to_generate - 1:
            logging.info(f"Waiting {API_DELAY_SECONDS}s before next page...")
            time.sleep(API_DELAY_SECONDS)

    # 4. Completion Summary (moved inside generate_book)
    logging.info("\n--- Visual Story Generation Process Complete ---")
    logging.info(f"Successfully generated images for {successful_pages} out of {total_pages_to_generate} pages.")
    if successful_pages > 0:
        logging.info(f"Generated pages saved in folder: '{os.path.abspath(OUTPUT_FOLDER)}'")
    if successful_pages < total_pages_to_generate:
        logging.warning("Some page images could not be generated. Check logs above for specific errors.")
    elif total_pages_to_generate > 0:
        logging.info("All requested page images generated successfully!")
    else:
        # This case should be caught earlier (empty book_pages)
        logging.info("No pages were processed.")

    return True # Indicate the overall process ran


# --- Main Execution (Now uses generate_book) ---
def main():
    """ Main function to get user input and drive the visual story generation process. """
    logging.info("--- Interactive Visual Story Generator ---")

    # 1. Get user input
    story_idea = input("Enter the core idea or theme for your story (e.g., 'a space pirate searching for lost treasure'): ")
    if not story_idea:
        logging.error("No story idea provided. Exiting.")
        return

    user_style = input("Enter the desired visual theme or art style (e.g., 'Studio Ghibli anime', 'pixel art', 'watercolor'): ")
    # No need to provide a default here, generate_book handles empty style

    while True:
        try:
            user_pages_str = input(f"Enter the desired number of pages (e.g., 6): ")
            user_pages = int(user_pages_str)
            if user_pages > 0:
                 break
            else:
                 print("Please enter a positive number of pages.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # 2. Call the generation function
    success = generate_book(story_idea, user_style, user_pages)

    # 3. Optional: Provide final feedback based on return status
    if success:
        print("\nBook generation process finished. Check logs and the output folder.")
    else:
        print("\nBook generation failed during setup (e.g., story structure). Check logs for details.")


if __name__ == "__main__":
    main()