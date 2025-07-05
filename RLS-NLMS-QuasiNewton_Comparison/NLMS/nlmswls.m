clean_speech = load('clean_speech.txt'); % s(n)
noisy_speech = load('noisy_speech.txt'); % s(n) + v(n)
external_noise = load('external_noise.txt'); % w(n)

N = 8;
num_samples = length(noisy_speech);
w = zeros(N, 1);
y = zeros(num_samples, 1); 
output_signal = zeros(num_samples, 1); 
mu = 0.01; 
epsilon = 1e-5; 

% NLMS loop

delta = 1e-3;  

processing_times = zeros(num_samples - N + 1, 1);  

mu = 0.001;

delay_array = zeros(num_samples, 1);

delta = 1e-3; 

processing_times = zeros(num_samples - N + 1, 1);  

for n = N:num_samples
    x = external_noise(n:-1:n-N+1);  

    tic;  % Start timer
    
    y(n) = w' * x;  % Filter output
    e = noisy_speech(n) - y(n);  % Error
    output_signal(n) = e;

    % Weighting factor (inverse error magnitude)
    lambda = 1 / (abs(e) + delta);

    norm_factor = x' * x + epsilon;

    % WLS update
    w = w + mu * lambda * x * e / norm_factor;

    processing_times(n - N + 1) = toc;  % Record timing
end

avg_delay_sec = mean(processing_times);
fprintf('Average processing time per sample with WLS: %.6f seconds\n', avg_delay_sec);

snr_before = 10 * log10(sum(clean_speech.^2) / sum((noisy_speech - clean_speech).^2));
snr_after = 10 * log10(sum(clean_speech.^2) / sum((output_signal - clean_speech).^2));

% Display SNR improvement
disp(['SNR Before Noise Cancellation: ', num2str(snr_before), ' dB']);
disp(['SNR After Noise Cancellation: ', num2str(snr_after), ' dB']);

% Plot Results
figure;
subplot(5,1,1);
plot(clean_speech);
title('Clean Speech Signal'); xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(5,1,2);
plot(noisy_speech);
title('Noisy Speech Signal'); xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(5,1,3);
plot(y);
title('Filtered Output Signal (Estimated Noise)'); xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(5,1,4);
plot(output_signal);
title('Error Signal (Final Output)'); xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(5,1,5);
plot(w);
title('Adaptive Filter Weights'); xlabel('Samples'); ylabel('Amplitude'); grid on;

fs = 44100; 

clean_speech = clean_speech / max(abs(clean_speech));
noisy_speech = noisy_speech / max(abs(noisy_speech));
output_signal = output_signal / max(abs(output_signal));

output_signal = real(output_signal);

audiowrite('clean_speech.wav', clean_speech, fs);
audiowrite('noisy_speech.wav', noisy_speech, fs);
audiowrite('filtered_output.wav', output_signal, fs);

disp('Conversion complete: .txt files have been saved as .wav files.');

disp('Playing filtered output...');
filtered_audio_player = audioplayer(output_signal, fs);
playblocking(filtered_audio_player);

[correlation, lags] = xcorr(output_signal, clean_speech);

[~, max_idx] = max(abs(correlation));
estimated_total_delay = lags(max_idx);

disp(['Estimated Delay between Clean Speech and Filtered Output: ', num2str(estimated_total_delay), ' samples']);
