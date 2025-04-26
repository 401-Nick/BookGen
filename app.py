# app.py

from flask import Flask, render_template, request, jsonify
# Import the function from your new module
from visual_story_generator import generate_book

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the main index page with the form."""
    return render_template('index.html')

@app.route('/request_book', methods=['POST'])
def request_book_endpoint():
    """
    Receives book request data, calls the story generation logic,
    and returns the result or an error.
    """
    try:
        # 1. Get and validate input data from the request
        user_prompt = request.form.get('user_prompt')
        art_style = request.form.get('art_style')
        pages_str = request.form.get('pages')

        if not all([user_prompt, art_style, pages_str]):
            return jsonify({"error": "Missing required fields (user_prompt, art_style, pages)"}), 400

        try:
            pages = int(pages_str)
            if pages <= 0:
                raise ValueError("Number of pages must be positive.")
        except ValueError as e:
            return jsonify({"error": f"Invalid number of pages: {e}"}), 400

        # 2. Call the core logic function from the imported module
        print(f"Received request: Prompt='{user_prompt}', Style='{art_style}', Pages={pages}")
        print("Calling generate_book...")

        try:
            story_structure_result = generate_book(
                user_prompt=user_prompt,
                user_style=art_style,
                user_pages=pages
            )
            print("generate_book returned successfully.")

        except Exception as generation_error:
            print(f"Error during story generation: {generation_error}") # Log detailed error server-side
            return jsonify({"error": f"Failed to generate story structure: {str(generation_error)}"}), 500

        # 3. Format and return the successful response
        response_data = {
            "message": "Book structure generated successfully!",
            "request_details": {
                "user_prompt": user_prompt,
                "art_style": art_style,
                "pages": pages
            },
            "generated_structure": story_structure_result
        }
        return jsonify(response_data), 200

    except Exception as e:
        print(f"An unexpected error occurred in the endpoint: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)