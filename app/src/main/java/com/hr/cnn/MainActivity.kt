package com.hr.cnn

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.hr.cnn.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {

    private lateinit var imgIV: ImageView
    private lateinit var resTV: TextView
    private lateinit var scnBTN: Button
    private lateinit var camBTN: Button
    private lateinit var glryBTN: Button
    private lateinit var bitmap: Bitmap // Declare the bitmap as a class variable
    private lateinit var labels: Array<String>  // Declare the labels array

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        imgIV = findViewById(R.id.IV_pic)
        resTV = findViewById(R.id.TV_res)
        scnBTN = findViewById(R.id.BTN_enter)
        camBTN = findViewById(R.id.BTN_cam)
        glryBTN = findViewById(R.id.BTN_gallery)

        labels = Array(1001) { "" }  // Initialize the labels array
        loadLabels()

        scnBTN.setOnClickListener {
            try {
                val model = MobilenetV110224Quant.newInstance(this)

                // Creates inputs for reference.
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
                bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
                inputFeature0.loadBuffer(TensorImage.fromBitmap(bitmap).buffer)

                // Runs model inference and gets result.
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                val confidences = outputFeature0.floatArray
                val maxIdx = confidences.indices.maxByOrNull { confidences[it] } ?: -1

                if (maxIdx >= 0) {
                    val predictedLabel = labels[maxIdx]
                    val confidence = confidences[maxIdx] * 100 // Convert to percentage

                    // Display predicted name and accuracy
                    resTV.text = "Prediction: $predictedLabel\nConfidence: ${"%.2f".format(confidence)}%"
                } else {
                    resTV.text = "Unknown"
                }

                // Releases model resources if no longer used.
                model.close()

            } catch (e: IOException) {  // Catch the IOException
                e.printStackTrace()
            }
        }

        glryBTN.setOnClickListener {
            val intent = Intent()
            intent.action = Intent.ACTION_GET_CONTENT
            intent.type = "image/*"
            startActivityForResult(intent, 10)
        }

        camBTN.setOnClickListener {
            getCamPermission()
        }
    }

    private fun loadLabels() {
        try {
            val bufferedReader = BufferedReader(InputStreamReader(assets.open("labels_mob.txt")))
            var line: String? = bufferedReader.readLine()
            var count = 0
            while (line != null) {
                labels[count] = line
                count++
                line = bufferedReader.readLine()
            }
            bufferedReader.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun getMax(arr: FloatArray): Int {
        var maxIndex = 0
        for (i in arr.indices) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i
            }
        }
        return maxIndex
    }

    private fun getCamPermission() {
        // Check for camera permission on devices with API level >= 23 (Android 6.0)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                // If permission is not granted, request the permission
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 11)
            } else {
                // If permission is already granted, open the camera
                openCamera()
            }
        } else {
            // For devices with API level < 23, permission is automatically granted
            openCamera()
        }
    }

    private fun openCamera() {
        // Create an intent to open the camera
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, 14)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == 11) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // If the permission is granted, open the camera
                openCamera()
            } else {
                // If permission is denied, show a message to the user
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 10 && resultCode == RESULT_OK) { // Ensure resultCode is RESULT_OK
            data?.data?.let { uri ->  // Use safe call with 'let' to handle null cases
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)  // Get bitmap image
                    imgIV.setImageBitmap(bitmap)  // Set image in ImageView
                } catch (e: IOException) {  // Catch the IOException
                    e.printStackTrace()
                }
            }
        } else if (requestCode == 14 && resultCode == RESULT_OK) {
            bitmap = data?.extras?.get("data") as Bitmap // Retrieve the bitmap
            imgIV.setImageBitmap(bitmap)  // Set image in ImageView
        }
    }
}
