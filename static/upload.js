$(document).ready(function() {
	//取代預設form的submit動作
	$('form').on('submit', function(event) {
		
		event.preventDefault(); //取代預設form的submit動作

		var formData = new FormData($('form')[0]); //formData object

		$.ajax({
			xhr: function() {
				var xhr = new window.XMLHttpRequest();

				xhr.upload.addEventListener('progress', function(e) {

					if (e.lengthComputable) {

						console.log('Bytes Loaded: ' + e.loaded);
						console.log('Total Size: ' + e.total);
						console.log('Percentage Uploaded: ' + e.loaded / e.total)

						var percent = Math.round((e.loaded / e.total) * 100);
						//載入bootstrap進度條 progressBar是自定義的id aria-valuenow來自bootstrap
						$('progressBar').attr('aria-valuenow', percent).css('width', percent + '%').text(percent + '%')

					}
				});
			},
			type: 'POST',
			url: '/mergefile',
			data: formData,
			processData: false,
			contentType: false,
			success: function(){
				alert('上傳完成!');
			}
		});
	});
});