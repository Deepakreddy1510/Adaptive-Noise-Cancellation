% Adaptive Noise Cancellation using WLS Adaptive Filter

clean_speech = load('clean_speech.txt'); % s(n)
noisy_speech = load('noisy_speech.txt'); % s(n) + v(n)
external_noise = load('external_noise.txt'); % w(n)

N = 8; 
num_samples = length(noisy_speech);
w = zeros(N, 1); 
y = zeros(num_samples, 1); 
output_signal = zeros(num_samples, 1); 

rho = 0.995; 
epsilon = 1e-6;

R = epsilon * eye(N); 
p = zeros(N, 1); 

for n = N:num_samples
    x = external_noise(n:-1:n-N+1);
    y(n) = w' * x;
    e = noisy_speech(n) - y(n); 
    
    R = rho * R + x * x';    
    p = rho * p + x * noisy_speech(n); 

    w = (R + epsilon * eye(N)) \ p;
    
    output_signal(n) = e;
end

% Compute SNR
snr_before = 10 * log10(sum(clean_speech.^2) / sum((noisy_speech - clean_speech).^2));
snr_after = 10 * log10(sum(clean_speech.^2) / sum((output_signal - clean_speech).^2));

disp(['SNR Before Noise Cancellation: ', num2str(snr_before), ' dB']);
disp(['SNR After Noise Cancellation: ', num2str(snr_after), ' dB']);

% Plot
figure;
subplot(5,1,1); plot(clean_speech); title('Clean Speech Signal'); xlabel('Samples'); ylabel('Amplitude'); grid on;
subplot(5,1,2); plot(noisy_speech); title('Noisy Speech Signal'); xlabel('Samples'); ylabel('Amplitude'); grid on;
subplot(5,1,3); plot(y); title('Filtered Output Signal'); xlabel('Samples'); ylabel('Amplitude'); grid on;
subplot(5,1,4); plot(output_signal); title('Error Signal (Final Output)'); xlabel('Samples'); ylabel('Amplitude'); grid on;
subplot(5,1,5); plot(w); title('Adaptive Filter Weights'); xlabel('Samples'); ylabel('Amplitude'); grid on;

fs = 44100;
clean_speech = clean_speech / max(abs(clean_speech));
noisy_speech = noisy_speech / max(abs(noisy_speech));
output_signal = real(output_signal / max(abs(output_signal)));

audiowrite('clean_speech.wav', clean_speech, fs);
audiowrite('noisy_speech.wav', noisy_speech, fs);
audiowrite('filtered_output.wav', output_signal, fs);

disp('Conversion complete: .txt files have been saved as .wav files.');
disp('Playing filtered output...');
sound(output_signal, fs);
