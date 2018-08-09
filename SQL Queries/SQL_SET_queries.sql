UPDATE donations
SET converts = 1
FROM donations AS d
JOIN (SELECT 
  donor_id, MAX(donor_cart_sequence) as max_cart
FROM 
  donations
GROUP BY donor_id) AS foo ON d.donor_id = foo.donor_id
WHERE d.donor_cart_sequence < foo.max_cart; --NOTE: this took a long time


-- THIS DID NOT COMPLETE AFTER RUNNING FOR A DAY Set donors total donation amount
UPDATE donors
SET
	max_cart = (
		SELECT MAX(donor_cart_sequence)
		FROM donations
		WHERE donations.donor_id = donors.donor_id
		);