    function validateFile() {
      var fileInput = document.getElementById('image');
      var image= fileInput.files[0];
      var errorMessageSpan = document.getElementById('errorMessage');

      // Check if a file is selected
      if (!image) {
        errorMessageSpan.textContent = "Please select a file.";
        return false;
      }
    }
 