clear all
load('s1_trial_sequence_v1.mat')
load('s01.mat')
ind = find(eeg.movement_event);
mov_left_13_avg = 0;
for ind1 = 1:length(ind)-1
    mov_left_13 = eeg.movement_left(13,ind(ind1):ind(ind1+1));
    mov_left_13_avg = mov_left_13_avg + mov_left_13;
end
mov_left_13_avg = mov_left_13_avg./length(ind);
