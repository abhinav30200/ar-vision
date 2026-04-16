package com.example.glimpse

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Pipeline: YUV→RGBA → Rotate → Crop(1/zoom) → Lanczos4 upscale → Adaptive USM → [Edge overlay]
 *
 * Why Lanczos-4: sinc-based kernel preserves high-frequency edges (text) far
 * better than bilinear/bicubic which average and blur.
 *
 * Why Unsharp Mask: amplifies local contrast at edges lost during upscale.
 * Parameters scale with zoom — higher zoom = more aggressive sharpening.
 */
class GlimpseImageAnalyzer(
    private val onFrameProcessed: (Bitmap, Int) -> Unit
) : ImageAnalysis.Analyzer {

    @Volatile var zoomLevel = 1
    @Volatile var edgeEnhanceMode = false

    private var lastFpsTime = System.nanoTime()
    private var frameCount = 0
    private var currentFps = 0
    private var nv21Buffer: ByteArray? = null

    override fun analyze(image: ImageProxy) {
        try {
            frameCount++
            val now = System.nanoTime()
            if (now - lastFpsTime >= 1_000_000_000L) {
                currentFps = frameCount
                frameCount = 0
                lastFpsTime = now
            }

            val rgbaMat = convertToRgba(image)
            val rotated = rotateMat(rgbaMat, image.imageInfo.rotationDegrees)
            if (rotated !== rgbaMat) rgbaMat.release()

            val processed = applyZoomAndSharpen(rotated)
            if (processed !== rotated) rotated.release()

            if (edgeEnhanceMode) applyEdgeOverlay(processed)

            val bitmap = Bitmap.createBitmap(processed.cols(), processed.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(processed, bitmap)
            processed.release()

            onFrameProcessed(bitmap, currentFps)
        } catch (_: Exception) {
        } finally {
            image.close()
        }
    }

    /** Correct YUV_420_888 → RGBA conversion with proper pixel-stride handling. */
    private fun convertToRgba(image: ImageProxy): Mat {
        val w = image.width
        val h = image.height
        val sz = w * h * 3 / 2
        if (nv21Buffer == null || nv21Buffer!!.size != sz) nv21Buffer = ByteArray(sz)
        val nv21 = nv21Buffer!!

        val yPlane = image.planes[0]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer; yBuf.rewind()
        val vBuf = vPlane.buffer; vBuf.rewind()
        val yRowStride = yPlane.rowStride
        val uvRowStride = vPlane.rowStride
        val uvPixelStride = vPlane.pixelStride

        // Copy Y plane row-by-row (handles row-stride padding)
        var off = 0
        if (yRowStride == w) {
            yBuf.get(nv21, 0, w * h); off = w * h
        } else {
            for (row in 0 until h) { yBuf.position(row * yRowStride); yBuf.get(nv21, off, w); off += w }
        }

        // Copy VU interleaved from V plane (NV21 format)
        val uvH = h / 2; val uvW = w / 2
        if (uvPixelStride == 2) {
            // Common fast path: V buffer already has interleaved VU pairs
            for (row in 0 until uvH) {
                vBuf.position(row * uvRowStride)
                val len = if (row < uvH - 1) uvW * 2 else uvW * 2 - 1
                vBuf.get(nv21, off, len); off += uvW * 2
            }
        } else {
            val uBuf = image.planes[1].buffer; uBuf.rewind()
            for (row in 0 until uvH) {
                for (col in 0 until uvW) {
                    val idx = row * uvRowStride + col * uvPixelStride
                    nv21[off++] = vBuf.get(idx); nv21[off++] = uBuf.get(idx)
                }
            }
        }

        val yuvMat = Mat(h + h / 2, w, CvType.CV_8UC1)
        yuvMat.put(0, 0, nv21)
        val rgba = Mat()
        Imgproc.cvtColor(yuvMat, rgba, Imgproc.COLOR_YUV2RGBA_NV21)
        yuvMat.release()
        return rgba
    }

    private fun rotateMat(mat: Mat, deg: Int): Mat {
        if (deg == 0) return mat
        val r = Mat()
        when (deg) {
            90  -> Core.rotate(mat, r, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(mat, r, Core.ROTATE_180)
            270 -> Core.rotate(mat, r, Core.ROTATE_90_COUNTERCLOCKWISE)
            else -> return mat
        }
        return r
    }

    // Pre-built sharpening kernels (allocated once, reused every frame)
    // Kernel = identity + α * Laplacian  →  single filter2D replaces GaussianBlur + addWeighted
    //   [[ 0, -α,  0],  [-α, 1+4α, -α],  [ 0, -α,  0]]
    private val sharpenKernelLight: Mat by lazy {
        val a = 0.3; Mat(3, 3, CvType.CV_64FC1).apply {
            put(0, 0, 0.0, -a, 0.0, -a, 1.0+4*a, -a, 0.0, -a, 0.0)
        }
    }
    private val sharpenKernelMedium: Mat by lazy {
        val a = 0.5; Mat(3, 3, CvType.CV_64FC1).apply {
            put(0, 0, 0.0, -a, 0.0, -a, 1.0+4*a, -a, 0.0, -a, 0.0)
        }
    }
    private val sharpenKernelStrong: Mat by lazy {
        val a = 0.8; Mat(3, 3, CvType.CV_64FC1).apply {
            put(0, 0, 0.0, -a, 0.0, -a, 1.0+4*a, -a, 0.0, -a, 0.0)
        }
    }

    private fun applyZoomAndSharpen(input: Mat): Mat {
        val z = zoomLevel
        if (z <= 1) return input

        val w = input.cols(); val h = input.rows()
        val cw = w / z; val ch = h / z
        val cropped = Mat(input, Rect((w - cw) / 2, (h - ch) / 2, cw, ch))

        // INTER_LINEAR is ~16x faster than INTER_LANCZOS4 (2×2 vs 8×8 kernel).
        // Quality gap is closed by the sharpening pass below.
        val resized = Mat()
        Imgproc.resize(cropped, resized, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        cropped.release()

        // Single-pass Laplacian sharpen via filter2D (replaces 2-pass blur+addWeighted)
        val kernel = when { z <= 3 -> sharpenKernelLight; z <= 6 -> sharpenKernelMedium; else -> sharpenKernelStrong }
        Imgproc.filter2D(resized, resized, -1, kernel)
        return resized
    }

    private fun applyEdgeOverlay(mat: Mat) {
        val gray = Mat(); Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
        val edges = Mat(); Imgproc.Canny(gray, edges, 50.0, 150.0); gray.release()
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0, 2.0))
        Imgproc.dilate(edges, edges, k); k.release()
        mat.setTo(Scalar(255.0, 255.0, 0.0, 255.0), edges); edges.release()
    }
}
