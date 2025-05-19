import os
import time
import glob
import pandas as pd
from datetime import datetime

# Import the ImageToMarkdown class - adjust import path as needed
try:
    from parsepro.image_parser import ImageToMarkdown
except ImportError:
    print("Could not import ImageToMarkdown. Make sure parsepro is installed.")
    exit(1)

# Configuration
# =============================================
# Path to images folder
IMAGES_FOLDER = "docs/images"

# Single image URL for testing all providers
TEST_IMAGE_URL = "https://d2x3xhvgiqkx42.cloudfront.net/12345678-1234-1234-1234-1234567890ab/be0b50df-801c-4d6d-b81c-dc49333eccac/2023/01/06/a25dd6c7-8b3c-4e4a-875d-18d99286e144/9096cf8b-4ca5-423e-8794-ac16adfd5014.png"

# API Keys Mention your api keys here

API_KEYS = {
    "anthropic":"",
    "openai":"",
    "groq":"",
    "together":""
}

# Providers list
PROVIDERS = ["anthropic", "openai", "groq", "together"]

# Image formats to test
IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp", "gif"]

def run_test():
    """Run tests for all providers with different image formats and configurations."""
    print("=" * 60)
    print("ImageToMarkdown Comprehensive Test")
    print("=" * 60)
    
    # Initialize results dataframe
    results = []
    
    # 1. Test all providers with all image formats from local folder
    print("\nTesting local images for all providers and formats...")
    for provider in PROVIDERS:
        # Set the API key
        os.environ[f"{provider.upper()}_API_KEY"] = API_KEYS[provider]
        
        # Initialize parser
        try:
            img_parse = ImageToMarkdown(provider=provider)
            print(f"\nInitialized {provider} provider")
            
            # Test with different image formats
            for format in IMAGE_FORMATS:
                # Find an image of this format
                image_pattern = os.path.join(IMAGES_FOLDER, f"*.{format}")
                image_files = glob.glob(image_pattern)
                
                if not image_files:
                    print(f"❌ No {format} images found in {IMAGES_FOLDER}")
                    # Record the failure
                    results.append({
                        "Provider": provider,
                        "Image Type": format,
                        "Source": "Local",
                        "Parameters": "None",
                        "Success": False,
                        "Time (s)": 0,
                        "Output Length": 0,
                        "Error": f"No {format} images found"
                    })
                    continue
                
                # Use the first image of this format
                image_path = image_files[0]
                print(f"Testing {provider} with {format} image: {os.path.basename(image_path)}")
                
                try:
                    # Time the conversion
                    start_time = time.time()
                    
                    # Convert the image
                    output = img_parse.convert_image_to_markdown(image_path=image_path)
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    # Print preview of result
                    print(f"✅ Success! Time: {elapsed_time:.2f}s, Length: {len(output)} chars")
                    print(f"Preview: {output[:100]}...")
                    
                    # Save to file
                    output_file = f"output_{provider}_{format}.md"
                    with open(output_file, "w") as f:
                        f.write(output)
                    
                    # Record the success
                    results.append({
                        "Provider": provider,
                        "Image Type": format,
                        "Source": "Local",
                        "Parameters": "None",
                        "Success": True,
                        "Time (s)": round(elapsed_time, 2),
                        "Output Length": len(output),
                        "Error": ""
                    })
                    
                except Exception as e:
                    print(f"❌ Failed: {str(e)}")
                    # Record the failure
                    results.append({
                        "Provider": provider,
                        "Image Type": format,
                        "Source": "Local", 
                        "Parameters": "None",
                        "Success": False,
                        "Time (s)": 0,
                        "Output Length": 0,
                        "Error": str(e)
                    })
        
        except Exception as e:
            print(f"❌ Failed to initialize {provider} provider: {str(e)}")
            # Record the initialization failure for all formats
            for format in IMAGE_FORMATS:
                results.append({
                    "Provider": provider,
                    "Image Type": format,
                    "Source": "Local",
                    "Parameters": "None", 
                    "Success": False,
                    "Time (s)": 0,
                    "Output Length": 0,
                    "Error": f"Provider initialization failed: {str(e)}"
                })
    
    # 2. Test all providers with a single URL
    print("\nTesting image URL for all providers...")
    for provider in PROVIDERS:
        try:
            # Set the API key
            os.environ[f"{provider.upper()}_API_KEY"] = API_KEYS[provider]
            
            # Initialize parser
            img_parse = ImageToMarkdown(provider=provider)
            print(f"\nTesting {provider} with URL: {TEST_IMAGE_URL}")
            
            try:
                # Time the conversion
                start_time = time.time()
                
                # Convert the image
                output = img_parse.convert_image_to_markdown(image_url=TEST_IMAGE_URL)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Print preview of result
                print(f"✅ Success! Time: {elapsed_time:.2f}s, Length: {len(output)} chars")
                print(f"Preview: {output[:100]}...")
                
                # Save to file
                output_file = f"output_{provider}_url.md"
                with open(output_file, "w") as f:
                    f.write(output)
                
                # Record the success
                results.append({
                    "Provider": provider,
                    "Image Type": "URL",
                    "Source": TEST_IMAGE_URL,
                    "Parameters": "None",
                    "Success": True,
                    "Time (s)": round(elapsed_time, 2),
                    "Output Length": len(output),
                    "Error": ""
                })
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                # Record the failure
                results.append({
                    "Provider": provider,
                    "Image Type": "URL",
                    "Source": TEST_IMAGE_URL,
                    "Parameters": "None",
                    "Success": False,
                    "Time (s)": 0,
                    "Output Length": 0,
                    "Error": str(e)
                })
                
        except Exception as e:
            print(f"❌ Failed to initialize {provider}: {str(e)}")
            # Record the initialization failure
            results.append({
                "Provider": provider,
                "Image Type": "URL",
                "Source": TEST_IMAGE_URL,
                "Parameters": "None",
                "Success": False,
                "Time (s)": 0,
                "Output Length": 0,
                "Error": f"Provider initialization failed: {str(e)}"
            })
    
    # # 3. Test with both json and temperature parameters using a PNG image
    print("\nTesting with custom parameters (JSON and temperature)...")
    # Find a PNG image
    png_images = glob.glob(os.path.join(IMAGES_FOLDER, "*.png"))
    
    if not png_images:
        print(f"❌ No PNG images found in {IMAGES_FOLDER}")
    else:
        png_image = png_images[0]
        print(f"Using PNG image: {os.path.basename(png_image)}")
        
        for provider in PROVIDERS:
            try:
                # Set the API key
                os.environ[f"{provider.upper()}_API_KEY"] = API_KEYS[provider]
                
                # Initialize parser
                img_parse = ImageToMarkdown(provider=provider)
                
                # Use both JSON and temperature parameters
                kwargs = {"json": True, "temperature": 0.7}
                print(f"\nTesting {provider} with parameters: {kwargs}")
                
                try:
                    # Time the conversion
                    start_time = time.time()
                    
                    # Convert the image
                    output = img_parse.convert_image_to_markdown(image_path=png_image, kwargs=kwargs)
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    # Check if we got a JSON response
                    if isinstance(output, dict) and "markdown" in output:
                        markdown_output = output["markdown"]
                        output_type = "JSON"
                    else:
                        markdown_output = str(output)
                        output_type = "String (expected JSON)"
                    
                    # Print preview of result
                    print(f"✅ Success! Time: {elapsed_time:.2f}s, Output type: {output_type}")
                    print(f"Preview: {markdown_output[:100]}...")
                    
                    # Save to file
                    output_file = f"output_{provider}_params.md"
                    with open(output_file, "w") as f:
                        f.write(markdown_output)
                    
                    # Record the success
                    results.append({
                        "Provider": provider,
                        "Image Type": "png",
                        "Source": "Local",
                        "Parameters": "json=True, temp=0.7",
                        "Success": True,
                        "Time (s)": round(elapsed_time, 2),
                        "Output Length": len(markdown_output),
                        "Error": ""
                    })
                    
                except Exception as e:
                    print(f"❌ Failed: {str(e)}")
                    # Record the failure
                    results.append({
                        "Provider": provider,
                        "Image Type": "png",
                        "Source": "Local",
                        "Parameters": "json=True, temp=0.7",
                        "Success": False,
                        "Time (s)": 0,
                        "Output Length": 0,
                        "Error": str(e)
                    })
                    
            except Exception as e:
                print(f"❌ Failed to initialize {provider}: {str(e)}")
                # Record the initialization failure
                results.append({
                    "Provider": provider,
                    "Image Type": "png",
                    "Source": "Local",
                    "Parameters": "json=True, temp=0.7",
                    "Success": False,
                    "Time (s)": 0,
                    "Output Length": 0,
                    "Error": f"Provider initialization failed: {str(e)}"
                })
    
    # 4. Save results to Excel
    print("\nSaving results to Excel...")
    df = pd.DataFrame(results)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f"image_to_markdown_results_{timestamp}.xlsx"
    
    # Save to Excel
    df.to_excel(excel_file, index=False)
    print(f"✅ Results saved to {excel_file}")
    
    # Print summary
    successful_tests = df["Success"].sum()
    total_tests = len(df)
    print(f"\nTest Summary: {successful_tests}/{total_tests} tests passed ({successful_tests/total_tests*100:.1f}%)")
    
    # Print provider-specific success rates
    print("\nProvider Success Rates:")
    provider_stats = df.groupby("Provider")["Success"].agg(["count", "sum"])
    provider_stats["rate"] = provider_stats["sum"] / provider_stats["count"] * 100
    
    for provider, stats in provider_stats.iterrows():
        print(f"  {provider}: {stats['sum']}/{stats['count']} ({stats['rate']:.1f}%)")

if __name__ == "__main__":
    run_test()