document.getElementById("upload-form").addEventListener("submit", async function (event) {
    event.preventDefault();

    const fileInput = document.getElementById("file");
    const resultDiv = document.getElementById("result");

    if (fileInput.files.length === 0) {
        alert("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Disable the submit button to prevent multiple uploads
    const submitButton = document.querySelector(".btn");
    submitButton.disabled = true;
    submitButton.innerText = "Uploading...";

    try {
        const response = await fetch("/predict_image", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();
        console.log("Server Response:", result); // Debugging log

        if (response.ok && result.emotion) {
            resultDiv.innerHTML = 
                `<p>Predicted Emotion: <strong>${result.emotion}</strong></p>`;
            resultDiv.classList.remove("hidden");
        } else {
            resultDiv.innerHTML = 
                `<p class="text-red-500">Error: ${result.error || "Unknown error occurred"}</p>`;
            resultDiv.classList.remove("hidden");
        }
    } catch (error) {
        console.error("Error:", error);
        resultDiv.innerHTML = 
            `<p class="text-red-500">An error occurred. Please try again.</p>`;
        resultDiv.classList.remove("hidden");
    } finally {
        // Re-enable the button after the request is complete
        submitButton.disabled = false;
        submitButton.innerText = "Upload";
    }
});
