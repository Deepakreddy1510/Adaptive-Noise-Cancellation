clean_speech = load('clean_speech.txt');     % s(n)
noisy_speech = load('noisy_speech.txt');     % s(n) + v(n)
external_noise = load('external_noise.txt'); % w(n)

N = 8;                     
lambda = 0.9995;        
delta = 1e2;               
num_samples = length(noisy_speech);

wls_weights = linspace(0.5, 1.5, num_samples)';  

w = zeros(N, 1);           
P = eye(N) * delta;             
y = zeros(num_samples, 1);     
output_signal = zeros(num_samples, 1);  

% WLS-enhanced RLS loop
for n = N:num_samples
    x = external_noise(n:-1:n-N+1);   
    y(n) = w' * x;                 
    e = noisy_speech(n) - y(n);     
    
    weight = wls_weights(n);           
    weighted_error = weight * e;    
    output_signal(n) = weighted_error;

    % Gain vector
    Pi_x = P * x;
    g = Pi_x / (lambda + x' * Pi_x + 1e-6);

    w = w + g * weighted_error;

    P = (P - g * x' * P) / lambda;
end

% SNR Calculations
snr_before = 10 * log10(sum(clean_speech.^2) / sum((noisy_speech - clean_speech).^2));
snr_after = 10 * log10(sum(clean_speech.^2) / sum((output_signal - clean_speech).^2));

disp(['SNR Before WLS: ', num2str(snr_before), ' dB']);
disp(['SNR After WLS: ', num2str(snr_after), ' dB']);

figure;
subplot(5,1,1); plot(clean_speech); title('Clean Speech'); ylabel('Amplitude'); grid on;
subplot(5,1,2); plot(noisy_speech); title('Noisy Speech'); ylabel('Amplitude'); grid on;
subplot(5,1,3); plot(y); title('Estimated Noise'); ylabel('Amplitude'); grid on;
subplot(5,1,4); plot(output_signal); title('WLS Output'); ylabel('Amplitude'); grid on;
subplot(5,1,5); plot(w); title('Filter Weights'); xlabel('Samples'); ylabel('Amplitude'); grid on;

fs = 44100;
clean_speech = clean_speech / max(abs(clean_speech));
noisy_speech = noisy_speech / max(abs(noisy_speech));
output_signal = output_signal / max(abs(output_signal));

audiowrite('clean_speech.wav', clean_speech, fs);
audiowrite('noisy_speech.wav', noisy_speech, fs);
audiowrite('wls_filtered_output.wav', output_signal, fs);

disp('Playing WLS filtered output...');
sound(output_signal, fs);
