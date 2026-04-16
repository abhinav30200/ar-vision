package com.example.glimpse

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.glimpse.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageAnalyzer: GlimpseImageAnalyzer? = null
    private var currentZoom = 1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV init failed!")
        }

        if (allPermissionsGranted()) startCamera()
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQ_CODE)

        setupUI()
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun setupUI() {
        binding.btnZoomIn.setOnClickListener {
            if (currentZoom < 10) { currentZoom++; updateZoom() }
        }
        binding.btnZoomOut.setOnClickListener {
            if (currentZoom > 1) { currentZoom--; updateZoom() }
        }
        binding.toggleEdgeMode.setOnCheckedChangeListener { _, on ->
            imageAnalyzer?.edgeEnhanceMode = on
        }
    }

    private fun updateZoom() {
        binding.zoomLabel.text = "${currentZoom}x"
        imageAnalyzer?.zoomLevel = currentZoom
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()

            // Request 720p for good quality/performance balance
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalyzer = GlimpseImageAnalyzer { bitmap, fps ->
                runOnUiThread {
                    binding.processedView.setImageBitmap(bitmap)
                    binding.fpsCounter.text = "FPS: $fps"
                }
            }
            analysis.setAnalyzer(cameraExecutor, imageAnalyzer!!)

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == REQ_CODE) {
            if (allPermissionsGranted()) startCamera()
            else { Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show(); finish() }
        }
    }

    override fun onDestroy() { super.onDestroy(); cameraExecutor.shutdown() }

    companion object {
        private const val TAG = "GlimpseApp"
        private const val REQ_CODE = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
