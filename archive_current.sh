today=`date --iso-8601`
name=paradiseo_$today
git archive --prefix=$name/ --format zip master > $name.zip
echo $name.zip

