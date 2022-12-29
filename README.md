```python
# 카메라 프레임별 확인
import imageio
import matplotlib.pyplot as plt
fig, axis = plt.subplots(1, 10, figsize=(30,300))
for i in range(10):
  _frame = _frames[i]
  _fname = os.path.join(_datadir, _fname['file_path'] + '.png')
  img = imageio.imread(_fname)
  axis[i].imshow(img)
```

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.title('input')
plt.imshow(inputs[1,0,:,:].cpu().detach().numpy(),cmap='gray')
plt.subplot(1,3,2)
plt.title('output')
plt.imshow(outputs[1,1,:,:].cpu().detach().numpy(),cmap='gray')
plt.subplot(1,3,3)
plt.title('label')
plt.imshow(targets[1,1,:,:].cpu().detach().numpy(), cmap='gray')
plt.savefig('./_qcqcqc')
np.unique(outputs.cpu().detach().numpy())
```
```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.title('input')
plt.imshow(MRI_T1[0,:,:].detach().numpy(),cmap='hot')
plt.subplot(1,2,2)
plt.title('label')
plt.imshow(Mask[0,:,:].detach().numpy(), cmap='hot')
plt.savefig('./_qcqcqc')
```

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,6))
plt.subplot(1,4,1)
plt.title('Label')
plt.imshow(label[:,:,:,1].sum(axis=0), cmap='hot')
plt.subplot(1,4,2)
plt.title('Prediction')
plt.imshow(mask3d[:,:,:,1].sum(axis=0), cmap='hot')
plt.subplot(1,4,3)
plt.imshow(dilated1[:,:,:].sum(axis=0), cmap='hot')
plt.title('Prediction_dilation')
plt.subplot(1,4,4)
plt.title('Prediction_dilation_erosion')
plt.imshow(dilated_erosion1[:,:,:].sum(axis=0), cmap='hot')
plt.savefig('./_qcqcqc')
```

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.imshow(ori_targets[0,2,:,:,:].sum(axis=1).cpu(), cmap='hot')
plt.subplot(1,3,2)
final_output[final_output!=2]=0
plt.imshow(final_output.sum(axis=1).cpu(), cmap='hot')
plt.savefig('./_qcqcqc')
```

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.imshow(data_dict[0]['image'].get_array()[0,30, :,:], cmap='gray')
plt.subplot(1,3,2)
plt.imshow(data_dict[0]['label'].get_array()[0,30, :,:], cmap='gray')
plt.subplot(1,3,3)
plt.imshow(data_dict[0]['pseudo_artery'].get_array()[0,30, :,:], cmap='gray')
plt.savefig('./_qcqcqc')


```

```python
# hiera2 check
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,6))
plt.subplot(1,9,1)
plt.title('BG', fontsize=15)
plt.imshow(outputs.get_array()[0,0,75,:,:], cmap='gray')
plt.subplot(1,9,2)
plt.title('SA_LGEA', fontsize=15)
plt.imshow(outputs.get_array()[0,1,75,:,:], cmap='gray')
plt.subplot(1,9,3)
plt.title('RGEA', fontsize=15)
plt.imshow(outputs.get_array()[0,2,75,:,:], cmap='gray')
plt.subplot(1,9,4)
plt.title('PHA_RHA_LHA', fontsize=15)
plt.imshow(outputs.get_array()[0,3,75,:,:], cmap='gray')
plt.subplot(1,9,5)
plt.title('LGA', fontsize=15)
plt.imshow(outputs.get_array()[0,4,75,:,:], cmap='gray')
plt.subplot(1,9,6)
plt.title('GDA', fontsize=15)
plt.imshow(outputs.get_array()[0,5,75,:,:], cmap='gray')
plt.subplot(1,9,7)
plt.title('CT', fontsize=15)
plt.imshow(outputs.get_array()[0,6,75,:,:], cmap='gray')
plt.subplot(1,9,8)
plt.title('CHA', fontsize=15)
plt.imshow(outputs.get_array()[0,7,75,:,:], cmap='gray')
plt.subplot(1,9,9)
plt.title('Aorta', fontsize=15)
plt.imshow(outputs.get_array()[0,8,75,:,:], cmap='gray')
plt.savefig('./_hiera2_targets')
plt.subplot(1,3,1)
plt.title('middle node BG', fontsize=15)
plt.imshow(outputs.get_array()[0,9,75,:,:], cmap='gray')
plt.subplot(1,3,2)
plt.title('Aorta,CT,SA_LGEA,LGA,CHA,PHA_RHA_LHA', fontsize=15)
plt.imshow(outputs.get_array()[0,10,75,:,:], cmap='gray')
plt.subplot(1,3,3)
plt.title('GDA,RGEA', fontsize=15)
plt.imshow(outputs.get_array()[0,11,75,:,:], cmap='gray')
plt.savefig('./_hiera2_mid')
plt.subplot(1,2,1)
plt.title('top node BG', fontsize=15)
plt.imshow(outputs.get_array()[0,12,75,:,:], cmap='gray')
plt.subplot(1,2,2)
plt.title('Aorta,CT,SA_LGEA,LGA,CHA,PHA_RHA_LHA,GDA,RGEA(FG)', fontsize=15)
plt.imshow(outputs.get_array()[0,13,75,:,:], cmap='gray')
plt.savefig('./_hiera2_top')
```

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
plt.subplot(5,3,1)
plt.imshow(img_npy[0, 100, :, :], cmap='gray')
plt.title('input')
plt.subplot(5,3,2)
plt.imshow(img3_npy[0, 100, :, :], cmap='gray')
plt.title('PB label')
plt.subplot(5,3,3)
plt.imshow(seg_npy_onehot[0, 100, :, :], cmap='gray')
plt.title('label')

plt.subplot(5,3,4)
plt.imshow(data_dict[0]['image'][0,100,:,:], cmap='gray')
plt.title('CT patch1')
plt.subplot(5,3,7)
plt.imshow(data_dict[1]['image'][0,100,:,:], cmap='gray')
plt.title('CT patch2')
plt.subplot(5,3,10)
plt.imshow(data_dict[2]['image'][0,100,:,:], cmap='gray')
plt.title('CT patch3')
plt.subplot(5,3,13)
plt.imshow(data_dict[3]['image'][0,100,:,:], cmap='gray')
plt.title('CT patch4')

plt.subplot(5,3,5)
plt.imshow(data_dict[0]['BB_CT'][0,100,:,:], cmap='gray')
plt.title('BB_CT patch1')
plt.subplot(5,3,8)
plt.imshow(data_dict[1]['BB_CT'][0,100,:,:], cmap='gray')
plt.title('BB_CT patch2')
plt.subplot(5,3,11)
plt.imshow(data_dict[2]['BB_CT'][0,100,:,:], cmap='gray')
plt.title('BB_CT patch3')
plt.subplot(5,3,14)
plt.imshow(data_dict[3]['BB_CT'][0,100,:,:], cmap='gray')
plt.title('BB_CT patch4')

plt.subplot(5,3,6)
plt.imshow(data_dict[0]['label'][0,100,:,:], cmap='gray')
plt.title('label patch1')
plt.subplot(5,3,9)
plt.imshow(data_dict[1]['label'][0,100,:,:], cmap='gray')
plt.title('label patch2')
plt.subplot(5,3,12)
plt.imshow(data_dict[2]['label'][0,100,:,:], cmap='gray')
plt.title('label patch3')
plt.subplot(5,3,15)
plt.imshow(data_dict[3]['label'][0,100,:,:], cmap='gray')
plt.title('label patch4')
plt.tight_layout()

plt.savefig('./_qc')
```
```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
plt.subplot(5,3,1)
plt.imshow(img_npy[0, 100, :, :], cmap='gray')
plt.title('input')
plt.subplot(5,3,2)
plt.imshow(img2_npy[0, 100, :, :], cmap='gray')
plt.title('PB label')
plt.subplot(5,3,3)
plt.imshow(seg_npy_onehot[0, 100, :, :], cmap='gray')
plt.title('label')

plt.subplot(5,3,4)
plt.imshow(data_dict[0]['image'][0,100,:,:], cmap='gray')
plt.title('CT patch1')
plt.subplot(5,3,7)
plt.imshow(data_dict[1]['image'][0,100,:,:], cmap='gray')
plt.title('CT patch2')
plt.subplot(5,3,10)
plt.imshow(data_dict[2]['image'][0,100,:,:], cmap='gray')
plt.title('CT patch3')
plt.subplot(5,3,13)
plt.imshow(data_dict[3]['image'][0,100,:,:], cmap='gray')
plt.title('CT patch4')

plt.subplot(5,3,5)
plt.imshow(data_dict[0]['Whole_Artery_Aorta_subtract_PB_labels'][0,100,:,:], cmap='gray')
plt.title('PB label patch1')
plt.subplot(5,3,8)
plt.imshow(data_dict[1]['Whole_Artery_Aorta_subtract_PB_labels'][0,100,:,:], cmap='gray')
plt.title('PB label patch2')
plt.subplot(5,3,11)
plt.imshow(data_dict[2]['Whole_Artery_Aorta_subtract_PB_labels'][0,100,:,:], cmap='gray')
plt.title('PB label patch3')
plt.subplot(5,3,14)
plt.imshow(data_dict[3]['Whole_Artery_Aorta_subtract_PB_labels'][0,100,:,:], cmap='gray')
plt.title('PB label patch4')

plt.subplot(5,3,6)
plt.imshow(data_dict[0]['label'][0,100,:,:], cmap='gray')
plt.title('label patch1')
plt.subplot(5,3,9)
plt.imshow(data_dict[1]['label'][0,100,:,:], cmap='gray')
plt.title('label patch2')
plt.subplot(5,3,12)
plt.imshow(data_dict[2]['label'][0,100,:,:], cmap='gray')
plt.title('label patch3')
plt.subplot(5,3,15)
plt.imshow(data_dict[3]['label'][0,100,:,:], cmap='gray')
plt.title('label patch4')
plt.tight_layout()

plt.savefig('./_qc')
```

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.title('input', fontsize=15)
plt.imshow(inputs.get_array()[0,0,0,:,:], cmap='gray')
plt.subplot(1,2,2)
plt.title('output recon', fontsize=15)
plt.imshow(outputs_recon.get_array()[0,0,0,:,:], cmap='gray')
plt.savefig('./_qc')
```
