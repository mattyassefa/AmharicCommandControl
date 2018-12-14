package com.nlp.matty.amhariccommandcontrol;

import android.Manifest;
import android.bluetooth.BluetoothAdapter;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.Camera.Parameters;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.AudioManager;
import android.net.Uri;
import android.app.admin.DevicePolicyManager;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.speech.SpeechRecognizer;
//import android.support.design.widget.FloatingActionButton;
//import android.support.design.widget.Snackbar;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;


public class MainActivity extends AppCompatActivity {

    private static final int SAMPLE_RATE = 16000;
    private static final int SAMPLE_DURATION_MS = 1000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
    private static final long AVERAGE_WINDOW_DURATION_MS = 500;
    private static final float DETECTION_THRESHOLD = 0.70f;
    private static final int SUPPRESSION_MS = 1500;
    private static final int MINIMUM_COUNT = 3;
    private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30;
    private static final String LABEL_FILENAME = "file:///android_asset/conv_actions_labels.txt";
    private static final String MODEL_FILENAME = "file:///android_asset/conv_actions_frozen.pb";
    private static final String INPUT_DATA_NAME = "decoded_sample_data:0";
    private static final String SAMPLE_RATE_NAME = "decoded_sample_data:1";
    private static final String OUTPUT_SCORES_NAME = "labels_softmax";

    // UI elements.
    private static final int REQUEST_RECORD_AUDIO = 13;
    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    private final ReentrantLock recordingBufferLock = new ReentrantLock();
    // Working variables.
    short[] recordingBuffer = new short[RECORDING_LENGTH];
    int recordingOffset = 0;
    boolean shouldContinue = true;
    boolean shouldContinueRecognition = true;
    private Button quitButton;
    private ListView labelsListView;
    private Thread recordingThread;
    private Thread recognitionThread;
    private TensorFlowInferenceInterface inferenceInterface;
    private List<String> labels = new ArrayList<String>();
    private List<String> displayedLabels = new ArrayList<String>();
    private RecognizeCommands recognizeCommands = null;

    private TextView label;
    boolean speakOff = true;
    FloatingActionButton fabSpk;

    private SpeechRecognizer contactsRecognizer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        label = findViewById(R.id.label);
        fabSpk = findViewById(R.id.fabSpeak);
    }

    public void recognition() {
        // Load the labels for the model, but only display those that don't start
        // with an underscore.
        String actualFilename = LABEL_FILENAME.split("file:///android_asset/")[1];
        Log.i(LOG_TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
                if (line.charAt(0) != '_') {
                    displayedLabels.add(line.substring(0, 1).toUpperCase() + line.substring(1));
                }
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }


        // Set up an object to smooth recognition results to increase accuracy.
        recognizeCommands =
                new RecognizeCommands(
                        labels,
                        AVERAGE_WINDOW_DURATION_MS,
                        DETECTION_THRESHOLD,
                        SUPPRESSION_MS,
                        MINIMUM_COUNT,
                        MINIMUM_TIME_BETWEEN_SAMPLES_MS);

        // Load the TensorFlow model.
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILENAME);

        // Start the recording and recognition threads.
        requestMicrophonePermission();
        startRecording();
        startRecognition();
    }

    private void requestMicrophonePermission() {
        ActivityCompat.requestPermissions(MainActivity.this,
                new String[]{android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_RECORD_AUDIO
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording();
            startRecognition();
        }
    }

    public synchronized void startRecording() {
        if (recordingThread != null) {
            return;
        }
        shouldContinue = true;
        recordingThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                record();
                            }
                        });
        recordingThread.start();
    }

    public synchronized void stopRecording() {
        if (recordingThread == null) {
            return;
        }
        shouldContinue = false;
        recordingThread = null;
    }

    private void record() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        // Estimate the buffer size we'll need for this device.
        int bufferSize =
                AudioRecord.getMinBufferSize(
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;
        }
        short[] audioBuffer = new short[bufferSize / 2];

        AudioRecord record =
                new AudioRecord(
                        MediaRecorder.AudioSource.DEFAULT,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!");
            return;
        }

        record.startRecording();

        Log.v(LOG_TAG, "Start recording");

        // Loop, gathering audio data and copying it to a round-robin buffer.
        while (shouldContinue) {
            int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
            int maxLength = recordingBuffer.length;
            int newRecordingOffset = recordingOffset + numberRead;
            int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
            int firstCopyLength = numberRead - secondCopyLength;
            // We store off all the data for the recognition thread to access. The ML
            // thread will copy out of this buffer into its own, while holding the
            // lock, so this should be thread safe.
            recordingBufferLock.lock();
            try {
                System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
                System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
                recordingOffset = newRecordingOffset % maxLength;
            } finally {
                recordingBufferLock.unlock();
            }
        }

        record.stop();
        record.release();
    }

    public synchronized void startRecognition() {
        if (recognitionThread != null) {
            return;
        }
        shouldContinueRecognition = true;
        recognitionThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                recognize();
                            }
                        });
        recognitionThread.start();
    }

    public synchronized void stopRecognition() {
        if (recognitionThread == null) {
            return;
        }
        shouldContinueRecognition = false;
        recognitionThread = null;
    }

    private void recognize() {
        Log.v(LOG_TAG, "Start recognition");

        short[] inputBuffer = new short[RECORDING_LENGTH];
        float[] floatInputBuffer = new float[RECORDING_LENGTH];
        float[] outputScores = new float[labels.size()];
        String[] outputScoresNames = new String[]{OUTPUT_SCORES_NAME};
        int[] sampleRateList = new int[]{SAMPLE_RATE};

        // Loop, grabbing recorded data and running the recognition model on it.
        while (shouldContinueRecognition) {
            // The recording thread places data in this round-robin buffer, so lock to
            // make sure there's no writing happening and then copy it to our own
            // local version.
            recordingBufferLock.lock();
            try {
                int maxLength = recordingBuffer.length;
                int firstCopyLength = maxLength - recordingOffset;
                int secondCopyLength = recordingOffset;
                System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
                System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
            } finally {
                recordingBufferLock.unlock();
            }

            // We need to feed in float values between -1.0f and 1.0f, so divide the
            // signed 16-bit inputs.
            for (int i = 0; i < RECORDING_LENGTH; ++i) {
                floatInputBuffer[i] = inputBuffer[i] / 32767.0f;
            }

            // Run the model.
            inferenceInterface.feed(SAMPLE_RATE_NAME, sampleRateList);
            inferenceInterface.feed(INPUT_DATA_NAME, floatInputBuffer, RECORDING_LENGTH, 1);
            inferenceInterface.run(outputScoresNames);
            inferenceInterface.fetch(OUTPUT_SCORES_NAME, outputScores);

            // Use the smoother to figure out if we've had a real recognition event.
            long currentTime = System.currentTimeMillis();
            final RecognizeCommands.RecognitionResult result = recognizeCommands.processLatestResults(outputScores, currentTime);

            runOnUiThread(
                    new Runnable() {
                        @Override
                        public void run() {
                            // If we do have a new command, highlight the right list entry.
                            if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
                                int labelIndex = -1;
                                for (int i = 0; i < labels.size(); ++i) {
                                    if (labels.get(i).equals(result.foundCommand)) {
                                        labelIndex = i;
                                    }
                                }
                                //label.setText(result.foundCommand);
                                if (result.foundCommand.equals("birhane_chemir"))
                                    brightnessUp();
                                else if (result.foundCommand.equals("birhane_kenese"))
                                    brightnessDown();
                                else if (result.foundCommand.equals("bluetooth_abra"))
                                    bluetoothOn();
                                else if (result.foundCommand.equals("bluetooth_atfa"))
                                    bluetoothOff();
                                else if (result.foundCommand.equals("camera-kifete"))
                                    openCamera();
                                else if (result.foundCommand.equals("data_atfa"))
                                    dataOn();
                                else if (result.foundCommand.equals("data_atfa"))
                                    dataOff();
                                else if (result.foundCommand.equals("demtse_chemir"))
                                    volumeUp(); //volume up
                                else if (result.foundCommand.equals("demtse_kenese"))
                                    volumeDown(); //volume down
                                else if (result.foundCommand.equals(("dewele")))
                                    dial();
                                else if (result.foundCommand.equals("kolefe"))
                                    lock(); //lock phone
                                else if (result.foundCommand.equals("mebrat_abra"))
                                    flashlightOn();
                                else if (result.foundCommand.equals("mebrat_atfa"))
                                    flashlightOff();
                                else if (result.foundCommand.equals("meleket_kifete"))
                                    openMessages();
                                else if (result.foundCommand.equals("wifi_abra"))
                                    wifiOn();
                                else if (result.foundCommand.equals("wifi_atfa"))
                                    wifiOff();
                                else if (result.foundCommand.equals("yeken_seleda_kifete"))
                                    openCalendar();
                                else if (result.foundCommand.equals("zero"))
                                    label.append("0");
                                else if (result.foundCommand.equals("ande"))
                                    label.append("1");
                                else if (result.foundCommand.equals("hulete"))
                                    label.append("2");
                                else if (result.foundCommand.equals("sosete"))
                                    label.append("3");
                                else if (result.foundCommand.equals("arate"))
                                    label.append("4");
                                else if (result.foundCommand.equals("ameste"))
                                    label.append("5");
                                else if (result.foundCommand.equals("sideste"))
                                    label.append("6");
                                else if (result.foundCommand.equals("sebate"))
                                    label.append("7");
                                else if (result.foundCommand.equals("semente"))
                                    label.append("8");
                                else if (result.foundCommand.equals("zetegne"))
                                    label.append("9");

/*                                final View labelView = labelsListView.getChildAt(labelIndex - 2);

                                AnimatorSet colorAnimation =
                                        (AnimatorSet)
                                                AnimatorInflater.loadAnimator(
                                                        MainActivity.this, R.animator.color_animation);
                                colorAnimation.setTarget(labelView);
                                colorAnimation.start();*/
                            }
                        }
                    });
            try {
                // We don't need to run too frequently, so snooze for a bit.
                Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
            } catch (InterruptedException e) {
                // Ignore
            }
        }

        Log.v(LOG_TAG, "End recognition");
    }

    public void onClickSpeak (View view){
        if (speakOff)
        {
            recognition();
            fabSpk.setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.ic_mic_off_black_24dp));
            Toast.makeText(this, "Speak now", Toast.LENGTH_LONG).show(); //ተናገር!
            speakOff = false;
        }
        else {
            stopRecognition();
            fabSpk.setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.ic_mic_black_24dp));
            speakOff = true;
        }

//        Button btn = (Button)findViewById(R.id.fabSpeak);
//        btn.setEnabled(false);
//        Button btn1 = (Button)findViewById(R.id.btnStop);
//        btn1.setEnabled(true);
    }

    public void onClickStop (View view){
        stopRecognition();
//        Button btn = (Button)findViewById(R.id.btnSpeak);
//        btn.setEnabled(true);
//        Button btn1 = (Button)findViewById(R.id.btnStop);
//        btn1.setEnabled(false);
    }

    //ON/OFF Functions
    public void wifiOn (){
        // WiFi ON
        try {
            WifiManager wifiManager = (WifiManager) this.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            if (wifiManager.isWifiEnabled())
                Toast.makeText(this, "WiFi is already ON", Toast.LENGTH_LONG).show();
            else
                wifiManager.setWifiEnabled(true);
        }
        catch (Exception e)
        {
            ;
        }
    }

    public void wifiOff (){
        // WiFi OFF
        try {
            WifiManager wifiManager = (WifiManager) this.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            if (!wifiManager.isWifiEnabled())
                Toast.makeText(this, "WiFi is already OFF", Toast.LENGTH_LONG).show();
            else
                wifiManager.setWifiEnabled(false);
        }
        catch (Exception e)
        {
            ;
        }
    }

    public void bluetoothOn (){
        // Bluetooth ON
        try {
            BluetoothAdapter mBluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
            if (mBluetoothAdapter.isEnabled())
                Toast.makeText(this, "Bluetooth is already ON", Toast.LENGTH_LONG).show();
            else
                mBluetoothAdapter.enable();
        }
        catch (Exception e)
        {
            ;
        }
    }

    public void bluetoothOff (){
        // Bluetooth OFF
        try {
            BluetoothAdapter mBluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
            if (!mBluetoothAdapter.isEnabled())
                Toast.makeText(this, "Bluetooth is already OFF", Toast.LENGTH_LONG).show();
            else
                mBluetoothAdapter.disable();
        }
        catch (Exception e)
        {
            ;
        }
    }

    public void dataOn (){
        // MobileData ON
        try {
            TelephonyManager telephonyService = (TelephonyManager) getSystemService(Context.TELEPHONY_SERVICE);

            Method setMobileDataEnabledMethod = telephonyService.getClass().getDeclaredMethod("setDataEnabled", boolean.class);

            if (null != setMobileDataEnabledMethod)
            {
                setMobileDataEnabledMethod.invoke(telephonyService, true);
            }
        }
        catch (Exception e)
        {
            ;
        }
    }

    public void dataOff (){
        // MobileData OFF
        try {
            TelephonyManager telephonyService = (TelephonyManager) getSystemService(Context.TELEPHONY_SERVICE);

            Method setMobileDataEnabledMethod = telephonyService.getClass().getDeclaredMethod("setDataEnabled", boolean.class);
            if (null != setMobileDataEnabledMethod)
            {
                setMobileDataEnabledMethod.invoke(telephonyService, false);
            }
        }
        catch (Exception e)
        {
            ;
        }
    }

    public void flashlightOn (){
        // Flashlight ON
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                try {
                    CameraManager camManager = (CameraManager) this.getSystemService(Context.CAMERA_SERVICE);
                    String cameraId = null; // Usually front camera is at 0 position.
                    if (camManager != null) {
                        cameraId = camManager.getCameraIdList()[0];
                        camManager.setTorchMode(cameraId, true);
                    }
                } catch (CameraAccessException e) {
                    ;//Log.e(TAG, e.toString());
                }
            } else {
                Camera mCamera = Camera.open();
                Parameters parameters = mCamera.getParameters();
                parameters.setFlashMode(Camera.Parameters.FLASH_MODE_TORCH);
                mCamera.setParameters(parameters);
                mCamera.startPreview();
            }
        }
        catch (Exception e)
        {
            ;
        }
    }

    public void flashlightOff () {
        // Flashlight OFF
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                try {
                    String cameraId;
                    CameraManager camManager = (CameraManager) this.getSystemService(Context.CAMERA_SERVICE);
                    if (camManager != null) {
                        cameraId = camManager.getCameraIdList()[0]; // Usually front camera is at 0 position.
                        camManager.setTorchMode(cameraId, false);
                    }
                } catch (CameraAccessException e) {
                    e.printStackTrace();
                }
            } else {
                Camera mCamera = Camera.open();
                Parameters parameters = mCamera.getParameters();
                parameters.setFlashMode(Camera.Parameters.FLASH_MODE_OFF);
                mCamera.setParameters(parameters);
                mCamera.stopPreview();
            }
        } catch (Exception e) {
            ;
        }
    }

    //UP/Down Functions
    public void brightnessUp (){
        // Increase Brightness
        try {
            //sets manual mode and brightnes 255
            Settings.System.putInt(getContentResolver(), Settings.System.SCREEN_BRIGHTNESS_MODE, Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL);  //this will set the manual mode (set the automatic mode off)
            Settings.System.putInt(getContentResolver(), Settings.System.SCREEN_BRIGHTNESS, 255);  //this will set the brightness to maximum (255)

            //refreshes the screen
            int br = Settings.System.getInt(getContentResolver(), Settings.System.SCREEN_BRIGHTNESS);
            WindowManager.LayoutParams lp = getWindow().getAttributes();
            lp.screenBrightness = (float) br / 255;
            getWindow().setAttributes(lp);
        }
        catch (Exception e)
        {
            ;
        }
    }

    public void brightnessDown (){
        // Increase Brightness
        try {
            //sets manual mode and brightnes 255
            Settings.System.putInt(getContentResolver(), Settings.System.SCREEN_BRIGHTNESS_MODE, Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL);  //this will set the manual mode (set the automatic mode off)
            Settings.System.putInt(getContentResolver(), Settings.System.SCREEN_BRIGHTNESS, 255);  //this will set the brightness to maximum (255)

            //refreshes the screen
            int br = Settings.System.getInt(getContentResolver(), Settings.System.SCREEN_BRIGHTNESS);
            WindowManager.LayoutParams lp = getWindow().getAttributes();
            lp.screenBrightness = (float) br / 255;
            getWindow().setAttributes(lp);
        }
        catch (Exception e)
        {
            ;
        }
    }

    //TODO
    public void volumeUp () {
        // Increase volume
        try {
            AudioManager audioManager = (AudioManager) getApplicationContext().getSystemService(Context.AUDIO_SERVICE);
            audioManager.adjustVolume(AudioManager.ADJUST_RAISE, AudioManager.FLAG_PLAY_SOUND);
        }
        catch (Exception e)
        {
            ;
        }
    }

    public void volumeDown() {
        // Decrease Volume
        try {
            AudioManager audioManager = (AudioManager) getApplicationContext().getSystemService(Context.AUDIO_SERVICE);
            audioManager.adjustVolume(AudioManager.ADJUST_LOWER, AudioManager.FLAG_PLAY_SOUND);
        }
        catch (Exception e)
        {
            ;
        }
    }

    //OPEN Functions
    public void openMessages() {
        // Redirect to messages
        Intent sendIntent = new Intent(Intent.ACTION_VIEW);
        sendIntent.setData(Uri.parse("sms:"));
        startActivity(sendIntent);
    }

    public void openCamera() {
        // Open Camera
        Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
        startActivity(intent);
    }

    public void openCalendar() {
        //open Calendar
    }

    //LOCK
    public void lock() {
        // Lock phone
//        WindowManager wm = Context.getSystemService(Context.WINDOW_SERVICE); //Get the context
//
//        Window window = getWindow();
//        window.addFlags(wm.LayoutParams.FLAG_DISMISS_KEYGUARD);  //Unlock the screen
//
        DevicePolicyManager mDPM = (DevicePolicyManager)getSystemService(Context.DEVICE_POLICY_SERVICE); //Lock the screen

//        KeyguardManager km = (KeyguardManager) getSystemService(Context.KEYGUARD_SERVICE);
//        final KeyguardManager.KeyguardLock kl = km.newKeyguardLock("MyKeyguardLock");
//        kl.disableKeyguard();
//
//        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
//        WakeLock wakeLock = pm.newWakeLock(PowerManager.FULL_WAKE_LOCK
//                | PowerManager.ACQUIRE_CAUSES_WAKEUP
//                | PowerManager.ON_AFTER_RELEASE, "MyWakeLock");
//        wakeLock.acquire();
//        wakeLock.release();
    }

    public void dial() {
        //dial
        String phoneNo = label.getText().toString();
        Intent intent = new Intent(Intent.ACTION_CALL, Uri.parse(phoneNo));
        if (ActivityCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.CALL_PHONE) == PackageManager.PERMISSION_GRANTED ) {
            startActivity(intent);
        }
    }
}
