clc; clear;

%  Mode Selection 
mode = 'partial'; 

clean_speech = load('clean_speech2.txt'); % s(n)
noisy_speech = load('noisy_speech_file2_time_varying_amplitude_f0_2kHz_f1_8243Hz.txt'); % s(n) + v(n)
external_noise = load('external_noise_file2_time_varying_amplitude_f0_2kHz_f1_8243Hz.txt'); % w(n)

N = 8;                         
fs = 44100;                    
num_samples = length(noisy_speech);

% LMS Parameters
w = zeros(N, 1);               % weights
y = zeros(num_samples, 1);     % output signal
output_signal = zeros(num_samples, 1);  % Error signal
H_inv = eye(N) * 1e-1;         % Inverse Hessian initialization
gamma = 0.91;                  % Forgetting factor
lambda_max = 1;                % Initial max eigenvalue estimate
epsilon = 1e-6;                % Regularization term
x_buffer = zeros(N, 1);        % Input buffer for LMS

% obtaining notch filter coeffs 
if strcmp(mode, 'partial')
    % tones given by user
    tonal_freqs = [2000];  
    r = 0.999;  
    % filter coeffs
    b_all = 1; a_all = 1;
    for k = 1:length(tonal_freqs)
        theta = 2*pi*tonal_freqs(k)/fs;
        b_k = [1, -2*cos(theta), 1];
        a_k = [1, -2*r*cos(theta), r^2];
        b_all = conv(b_all, b_k);
        a_all = conv(a_all, a_k);
    end
    notch_state = zeros(max(length(a_all), length(b_all))-1, 1);
end

for n = 1:num_samples
    %switch to select mode
    if strcmp(mode, 'partial')
        [filtered_noise, notch_state] = filter(b_all, a_all, external_noise(n), notch_state);
    else
        filtered_noise = external_noise(n);
    end

   
    x_buffer = [filtered_noise; x_buffer(1:end-1)];

    if n >= N
        y(n) = w' * x_buffer;
        output_signal(n) = noisy_speech(n) - y(n);

        % Hessian estimation
        H_est = gamma * H_inv + (1 - gamma) * (x_buffer * x_buffer') + epsilon * eye(N);
        lambda_max = max(lambda_max, trace(H_est) / N);
        lambda_max = max(lambda_max,20);
        u = 2/lambda_max;
        % u =min(2 /lambda_max,0.3);
        % Inverse Hessian update
        H_inv = gamma * H_inv + ...
                (1 - gamma) * (eye(N) - u * (x_buffer * x_buffer') / (1 + u * x_buffer' * x_buffer)) * H_inv;
        H_inv = (H_inv + H_inv') / 2 + epsilon * eye(N);

        % Adaptive filter update
        w = w + u * H_inv * x_buffer * output_signal(n);
    end
end

%Snr values calculation and playing audios
snr_before = 10 * log10(sum(clean_speech.^2) / sum((noisy_speech - clean_speech).^2));
snr_after = 10 * log10(sum(clean_speech.^2) / sum((output_signal - clean_speech).^2));
disp(['Mode of Operation: ', upper(mode)]);
snr_gain_adaptive = snr_after - snr_before;
disp(['SNR Before Noise Cancellation: ', num2str(snr_before), ' dB']);
disp(['SNR After Adaptive Cancellation: ', num2str(snr_after), ' dB']);
disp(['SNR Gain After Adaptive Cancellation: ', num2str(snr_gain_adaptive), ' dB']);

% clean_speech = clean_speech / max(abs(clean_speech));
% noisy_speech = noisy_speech / max(abs(noisy_speech));
% output_signal = real(output_signal / max(abs(output_signal)));

% audiowrite('clean_speech.wav', clean_speech, fs);
% audiowrite('noisy_speech.wav', noisy_speech, fs);
% audiowrite('filtered_output.wav', output_signal, fs);


disp('Audio saved as .wav files.');
sound(output_signal, fs);

%signal plots
figure;
subplot(5,1,1); plot(clean_speech); title('Clean Speech'); grid on;
subplot(5,1,2); plot(noisy_speech); title('Noisy Speech'); grid on;
subplot(5,1,3); plot(y); title('Noise Estimate (Adaptive Output)'); grid on;
subplot(5,1,4); plot(output_signal); title('Final Output Signal'); grid on;
subplot(5,1,5); plot(w); title('Adaptive Filter Weights'); grid on;


% Notch filter metrics
if strcmp(mode, 'partial')
    [Hn, f] = custom_freqz(b_all, a_all, 441000, fs);  % Full FFT for notch filter response
    figure;
    subplot(2,1,1); plot(f, 20*log10(abs(Hn))); title('Notch Filter Magnitude Response'); grid on;
    subplot(2,1,2); plot(f, unwrap(angle(Hn))*180/pi); title('Notch Filter Phase Response'); grid on;
    % Apply notch filter to the final output
output_signal_notched = filter(b_all, a_all, output_signal);
output_signal_notched = output_signal_notched / max(abs(output_signal_notched));

% new SNR as metric it shows how much non tonal retainment is there
snr_after_notched = 10 * log10(sum(clean_speech.^2) / sum((output_signal_notched - clean_speech).^2));

% Save and display
audiowrite('filtered_output_notched.wav', output_signal_notched, fs);
disp(['SNR After Output Notch Filtering: ', num2str(snr_after_notched), ' dB']);

% Plot comparison
figure;
subplot(3,1,1); plot(clean_speech); title('Clean Speech'); grid on;
subplot(3,1,2); plot(output_signal); title('Before Notch Filter'); grid on;
subplot(3,1,3); plot(output_signal_notched); title('After Notch Filter on Output'); grid on;
end

% Custom Frequency Response Function
function [H, w] = custom_freqz(b, a, n_points, fs)

    w = linspace(0, pi, n_points);  % Frequency grid (rad/sample)
    H = zeros(size(w));             % Frequency response (complex)

    for k = 1:length(w)
        ejw = exp(-1j * w(k) * (0:max(length(b), length(a)) - 1));
        num = sum(b(:).' .* ejw(1:length(b)));
        den = sum(a(:).' .* ejw(1:length(a)));
        H(k) = num / den;
    end

    w = w * fs / (2*pi); 
end

% FFT parameters
nfft = 2^nextpow2(length(noisy_speech));
f = linspace(0, fs/2, nfft/2);

% FFT of noisy speech
Y_noisy = fft(noisy_speech, nfft);
mag_noisy = abs(Y_noisy(1:nfft/2));

% FFT of filtered output
Y_filtered = fft(output_signal, nfft);
mag_filtered = abs(Y_filtered(1:nfft/2));

% Plot
figure;
plot(f, 20*log10(mag_noisy + eps), 'b', 'DisplayName', 'Noisy Speech'); hold on;
plot(f, 20*log10(mag_filtered + eps), 'r', 'DisplayName', 'Filtered Output');
title('FFT Magnitude Comparison');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
legend; grid on;
