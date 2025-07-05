clean_speech = load('clean_speech.txt'); % s(n)
noisy_speech = load('noisy_speech.txt'); % s(n) + v(n)
external_noise = load('external_noise.txt'); % w(n)

N = 8;
num_samples = length(noisy_speech);
w = zeros(N, 1);
y = zeros(num_samples, 1);
output_signal = zeros(num_samples, 1); 

H_inv = eye(N) * 1e-1; 
gamma = 0.91; 
lambda_max = 1; 
epsilon = 1e-6; 

% Log-Cosh parameters
delta_lc = 1e-6; % Small value for log-cosh to avoid instability for large errors

for n = N:num_samples
    x = external_noise(n:-1:n-N+1); 
    y(n) = w' * x; 
    
    % Log-Cosh cost function error
    e = noisy_speech(n) - y(n); 
    output_signal(n) = e; 

    % Log-Cosh cost function and gradient
    log_cosh = log(cosh(e) + delta_lc); 
    grad_log_cosh = tanh(e); 

    H_est = gamma * H_inv + (1 - gamma) * (x * x') + epsilon * eye(N);

    % Estimate max eigenvalue
    lambda_max = max(lambda_max, trace(H_est) / N); 
    lambda_max = max(lambda_max, 1e-3);

    u = 2 / lambda_max; 

    H_inv = gamma * H_inv + (1 - gamma) * (eye(N) - u * (x * x') / (1 + u * x' * x)) * H_inv;
    H_inv = (H_inv + H_inv') / 2 + epsilon * eye(N); 

    w = w + u * H_inv * x * grad_log_cosh;
end

snr_before = 10 * log10(sum(clean_speech.^2) / sum((noisy_speech - clean_speech).^2));
snr_after = 10 * log10(sum(clean_speech.^2) / sum((output_signal - clean_speech).^2));

disp(['SNR Before Noise Cancellation: ', num2str(snr_before), ' dB']);
disp(['SNR After Noise Cancellation: ', num2str(snr_after), ' dB']);

figure;
subplot(5,1,1);
plot(clean_speech);
title('Clean Speech Signal'); xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(5,1,2);
plot(noisy_speech);
title('Noisy Speech Signal'); xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(5,1,3);
plot(y);
title('Filtered Output Signal'); xlabel('Samples'); ylabel('Amplitude'); grid on;

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
